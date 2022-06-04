"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""


from logging import handlers

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFilter, Image
from pylab import *
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False


pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        # torch.cuda.empty_cache()
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

class ClassBlock(nn.Module):# classify the final result
    def __init__(self):
        super().__init__()
        self.c0 = nn.Linear(2048, 1024)
        self.dp0 = nn.Dropout(p=0.5)

        self.c1 = nn.Linear(1024, 512)
        self.dp1 = nn.Dropout(p=0.5)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.c2 = nn.Linear(512,512)
        self.dp2 = nn.Dropout(p=0.5)
        self.leakyrelu2 = nn.LeakyReLU(0.1)
        self.leakyrelu3 = nn.LeakyReLU(0.1)

        self.c3 = nn.Linear(512,1)
        self.re2 = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        h = self.dp0(x)
        h = self.c0(h)
        h = self.leakyrelu(h)

        h = self.dp1(h)
        h = self.c1(h)
        h = self.leakyrelu2(h)

        h = self.dp2(h)
        h = self.c2(h)
        h = self.leakyrelu3(h)

        h = self.c3(h)
        return h

def return_pytorch04_xception(pretrained=True):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = Xception(num_classes=1000)
    # pretrained = Meso4()
    if pretrained:
        # Load pretrained in torch 0.4+
        # pretrained.fc = pretrained.last_linear
        # del pretrained.last_linear
        state_dict = torch.load(
            '/4T/shengming/shengming/FaceForensics/weights/Xception/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        # pretrained.last_linear = pretrained.fc
        # del pretrained.fc
        model.fc = ClassBlock()
    return model


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)
        self.dp = nn.Dropout(p=0.2)
        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------
    def features_to_7(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return x

    def features_8_to_12(self, x):
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        return x

    def features_12_to_end(self,x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x


    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        y = x
        x = self.dp(x)
        x = self.last_linear(x)
        return y,x

    def forward(self, input):
        x = self.features(input)
        y,x = self.logits(x)
        return y,x



def get_xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        '''
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        '''

        model = Xception(num_classes=num_classes)
        state_dict = torch.load('/4T/shengming/shengming/FaceForensics/weights/Xception/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        # print(state_dict)
        model.load_state_dict(state_dict, False)

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model



def get_xcep_state_dict(pretrained_path='PretrainedModels/xception-b5690688.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    for name, weights in state_dict.items():
        #print(name)
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)

    state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s : %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt) #设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler() #往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(format_str) #设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

class MyGaussianBlur(ImageFilter.Filter):


    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)




class MixBlock(nn.Module):
    def __init__(self, c_in):
        super(MixBlock, self).__init__()
        self.A_query = nn.Conv2d(c_in, c_in, (1, 1))
        self.B_query = nn.Conv2d(c_in, c_in, (1, 1))

        self.A_key = nn.Conv2d(c_in, c_in, (1, 1))
        self.B_key = nn.Conv2d(c_in, c_in, (1, 1))

        self.softmax = nn.Softmax(dim=-1)

        self.A_gamma = nn.Parameter(torch.zeros(1))
        self.B_gamma = nn.Parameter(torch.zeros(1))

        self.A_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.A_bn = nn.BatchNorm2d(c_in)
        self.B_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.B_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_A, x_B):
        B, C, W, H = x_A.size()
        assert W == H

        q_A = self.A_query(x_A).view(-1, W, H)
        q_B = self.B_query(x_B).view(-1, W, H)
        M_query = torch.cat([q_A, q_B], dim=2)

        k_A = self.A_key(x_A).view(-1, W, H).transpose(1, 2)
        k_B = self.B_key(x_B).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_A, k_B], dim=1)

        energy = torch.bmm(M_query, M_key)
        attention = self.softmax(energy).view(B, C, W, W)

        att_B = x_B * attention * self.B_gamma
        y_A = x_A + self.A_bn(self.A_conv(att_B))

        att_A = x_A * attention * self.A_gamma
        y_B = x_B + self.B_bn(self.B_conv(att_A))

        return y_A, y_B

class MixBlock3D(nn.Module):
    def __init__(self, c_in):
        super(MixBlock3D, self).__init__()
        self.A_query = nn.Conv2d(c_in, c_in, (1, 1))
        self.B_query = nn.Conv2d(c_in, c_in, (1, 1))
        self.C_query = nn.Conv2d(c_in, c_in, (1, 1))

        self.A_key = nn.Conv2d(c_in, c_in, (1, 1))
        self.B_key = nn.Conv2d(c_in, c_in, (1, 1))
        self.C_key = nn.Conv2d(c_in, c_in, (1, 1))

        self.softmax = nn.Softmax(dim=-1)

        self.A_gamma = nn.Parameter(torch.zeros(1))
        self.B_gamma = nn.Parameter(torch.zeros(1))
        self.C_gamma = nn.Parameter(torch.zeros(1))

        self.A_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.A_bn = nn.BatchNorm2d(c_in)
        self.B_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.B_bn = nn.BatchNorm2d(c_in)
        self.C_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.C_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_A, x_B,x_C):
        B, C, W, H = x_A.size()
        assert W == H

        q_A = self.A_query(x_A).view(-1, W, H)
        q_B = self.B_query(x_B).view(-1, W, H)
        q_C = self.C_query(x_C).view(-1, W, H)
        M_query = torch.cat([q_A, q_B , q_C], dim=2) #[X,W,3H]

        k_A = self.A_key(x_A).view(-1, W, H).transpose(1, 2) #[X,H,W]
        k_B = self.B_key(x_B).view(-1, W, H).transpose(1, 2)
        k_C = self.C_key(x_C).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_A, k_B, k_C], dim=1) #[X,3H,W]

        energy = torch.bmm(M_query, M_key)
        attention = self.softmax(energy).view(B, C, W, W)

        att_for_a = (x_B * attention * self.B_gamma+ x_C * attention * self.C_gamma)*0.5
        y_A = x_A + self.A_bn(self.A_conv(att_for_a))

        att_for_b = (x_A * attention * self.A_gamma + x_C * attention * self.C_gamma)*0.5
        y_B = x_B + self.B_bn(self.B_conv(att_for_b))

        att_for_c = (x_A * attention * self.A_gamma + x_B * attention * self.B_gamma)*0.5
        y_C = x_C + self.C_bn(self.C_conv(att_for_c))

        return y_A, y_B,y_C

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Model Total Parameters Num: %.1fm , Trainable: %.1fm'%(total_num/1000000,trainable_num/1000000)}


def DCTFilter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def DCT_mat(size):
    return [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size)
              for j in range(size)] for i in range(size)]


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels,fontsize=10)
    plt.yticks(xlocations, labels,fontsize=10)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

labels = ['Really','Fake']
def draw(y_pred,y_true,title):
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm
    plt.figure(figsize=(3, 3), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if (c > 400):
            plt.text(x_val, y_val, c, color='white', fontsize=10, va='center', ha='center')
        else:
            plt.text(x_val, y_val, c, color='black', fontsize=10, va='center', ha='center')
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)

    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')

    plt.gcf().subplots_adjust(bottom=0.15)
    plot_confusion_matrix(cm_normalized, title='Confusion Matrix '+title)
    plt.show()




