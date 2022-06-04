
import random
import numpy as np
import torch
import visdom
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as trans
from efficientnet_pytorch import EfficientNet
from Utils import Logger, MyGaussianBlur, get_parameter_number, MixBlock, draw
from sklearn.metrics import precision_recall_curve, roc_curve, auc

logname='Log/20220604/'+os.path.basename(__file__)+'.log'

batch_size=8
epoch=10
CropSize=128
RandomCropNum=8 #随机剪裁数量
LR=1e-4
LRChange=0
log_fre=100
eval_frq=500
best = 0 #记录最优表现模型


Tstring='FFHQ-CelebraHQ-AFHQ'
Fstring='ProGAN-StyleGAN-StyleGAN2'

sa="Dataset"
TrainData=sa+'/Train.txt'



TACDTData= [sa+'/Test.txt',
            sa+'/CD-AFHQvsStyleGAN3.txt',
            sa+'/CD-CelebaHQvsStyleGAN3.txt',
            sa+'/CD-FFHQvsStyleGAN3.txt',
            sa+'/CD-MetafacesvsStyleGAN3.txt',
            sa+'/CD-LSUNvsStyleGAN3.txt',
            sa+'/CD-AFHQvsStyleGAN3-LQ.txt',
            sa+'/CD-CelebaHQvsStyleGAN3-LQ.txt',
            sa+'/CD-FFHQvsStyleGAN3-LQ.txt',
            sa+'/CD-MetafacesvsStyleGAN3-LQ.txt',
            sa+'/CD-LSUNvsStyleGAN3-LQ.txt',
            sa+'/CD-Global.txt'
            ]

str_ls=['Test',
        'AFHQ vs. StyleGAN3',
        'CelebaHQ vs. StyleGAN3',
        'FFHQ vs. StyleGAN3',
        'Metafaces vs. StyleGAN3',
        'LSUN vs. StyleGAN3',
        'AFHQ(LQ) vs. StyleGAN3(LQ)',
        'CelebaHQ(LQ) vs. StyleGAN3(LQ)',
        'FFHQ(LQ) vs. StyleGAN3(LQ)',
        'Metafaces(LQ) vs. StyleGAN3(LQ)',
        'LSUN(LQ) vs. StyleGAN3(LQ)',
        'Global'
        ]



def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(1) #设置随机种子

#装载训练数据
class Trainingdata(Dataset):
    def __init__(self):
        self.file = open(TrainData, encoding="utf-8").readlines()
        print("===训练数据共有："+str(len(self.file))+"条===")
        #+random.randint(-10,10)
        self.TenCrop = trans.Compose([
            trans.TenCrop(CropSize)
        ])

        self.RandomCrop = trans.Compose([
            trans.RandomCrop(CropSize)
        ])

        self.trans1=trans.RandomApply([
            trans.RandomRotation(90),
            trans.RandomAffine(90),
            trans.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
            trans.RandomGrayscale(p=0.1)
        ])

        self.trans2 = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):

        #划分标签
        if self.file[idx].strip('\n').split(" ")[1]=='T':
            v = 1
        else:
            v = 0

        img = Image.open(self.file[idx].strip('\n').split(" ")[0])

        if random.random() > 0.5: #随机模糊
            img = img.filter(MyGaussianBlur(radius=random.randint(1,3)))

        imglist=[]
        for i in range(RandomCropNum):
            imglist.append(self.trans2(self.trans1(self.RandomCrop(img).resize((CropSize, CropSize)))).cuda())

        #再添加一个FiveCrop剪裁
        TenCrop=self.TenCrop(img)
        for ii in range(len(TenCrop)):
            imglist.append(self.trans2(self.trans1(TenCrop[ii].resize((CropSize, CropSize)))).cuda())

        random.shuffle(imglist) #打乱顺序


        sample = {'img':[torch.cat((torch.cat(imglist[0:3], dim=2),
                                   torch.cat(imglist[3:6], dim=2),
                                   torch.cat(imglist[6:9], dim=2)), dim=1),
                         torch.cat((torch.cat(imglist[9:12], dim=2),
                                    torch.cat(imglist[12:15], dim=2),
                                    torch.cat(imglist[15:18], dim=2)), dim=1)],
                  'label': Variable(torch.tensor(v, requires_grad=False).to(torch.float32).cuda())}

        return sample

    def __len__(self):
        return len(self.file)

#装载测试集数据
class Testdata(Dataset):
    def __init__(self,index):
        self.file = open(TACDTData[index], encoding="utf-8").readlines()
        print("==="+str_ls[index]+" 测试数据共有：" + str(len(self.file)) + "条===")

        self.TenCrop = trans.Compose([
            trans.TenCrop(CropSize)
        ])
        self.RandomCrop = trans.Compose([
            trans.RandomCrop(CropSize)
        ])
        self.trans = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img = Image.open(self.file[idx].strip('\n').split(" ")[0])

        if self.file[idx].strip('\n').split(" ")[1] == 'T':
            v = 1
        else:
            v = 0

        imglist = []
        for i in range(RandomCropNum):
            imglist.append(self.trans(self.RandomCrop(img).resize((CropSize, CropSize))).cuda())

        TenCrop = self.TenCrop(img)
        for ii in range(len(TenCrop)):
            imglist.append(self.trans(TenCrop[ii].resize((CropSize, CropSize))).cuda())

        random.shuffle(imglist)  # 打乱顺序

        sample = {'img': [torch.cat((torch.cat(imglist[0:3], dim=2),
                                     torch.cat(imglist[3:6], dim=2),
                                     torch.cat(imglist[6:9], dim=2)), dim=1),
                          torch.cat((torch.cat(imglist[9:12], dim=2),
                                     torch.cat(imglist[12:15], dim=2),
                                     torch.cat(imglist[15:18], dim=2)), dim=1)],
                  'label': Variable(torch.tensor(v, requires_grad=False).to(torch.float32).cuda())}

        return sample

    def __len__(self):
        return len(self.file)



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.efficient1 = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3)
        self.efficient2 = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3)

        self.MixBlock = MixBlock(80)

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2560, 1)
        self.dp = nn.Dropout(p=0.2)

    def _norm_fea(self, fea):
        f = self.relu(fea)
        f = F.adaptive_avg_pool2d(f, (1, 1))
        f = f.view(f.size(0), -1)
        return f

    def forward(self, x):

        x0_fea = self.efficient1.feature_to_1(x[0]) # 80 24 24
        x1_fea = self.efficient2.feature_to_1(x[1])  # 80 24 24

        x0_fea, x1_fea = self.MixBlock.forward(x0_fea, x1_fea) #mix

        x0_fea = self.efficient1.feature_1_to_end(x0_fea) #1280 12 12
        x1_fea = self.efficient2.feature_1_to_end(x1_fea)  # 1280 12 12


        x0_fea = self._norm_fea(x0_fea)
        x1_fea= self._norm_fea(x1_fea)

        y = torch.cat((x0_fea, x1_fea), dim=1)
        f = self.dp(y)
        f = self.fc(f)

        return F.sigmoid(f).flatten()

def ROC_AUC(vis,Y,step):

    AUC=[]
    for i in range(len(str_ls)):

        if i==0:
            fpr, tpr, _ = roc_curve(Y[i][0], Y[i][1], pos_label=1)
            aucv = auc(fpr, tpr)  # 计算AUC
            vis.line(X=fpr, Y=tpr, win="ROC Step:" + str(step),
                     name=str_ls[i] + " (AUC=%.5f)" % aucv,
                     opts={'showlegend': True,
                           'title': "ROC Step:" + str(step),
                           'xlabel': "FPR",
                           'ylabel': "TPR",
                           })

            precision, recall, _ = precision_recall_curve(Y[i][0], Y[i][1])
            vis.line(X=recall, Y=precision, win="PR Step:" + str(step),
                     name = str_ls[i],
                     opts={'showlegend': True,
                           'title': "PR Step:" + str(step),
                           'xlabel': "Recall",
                           'ylabel': "Precision",
                           })
        else:
            fpr, tpr, _ = roc_curve(Y[i][0], Y[i][1], pos_label=1)
            aucv = auc(fpr, tpr)
            vis.line(X=fpr, Y=tpr, win="ROC Step:" + str(step),
                     name=str_ls[i] +" (AUC=%.5f)" % aucv,
                     update='append')


            precision, recall, _ = precision_recall_curve(Y[i][0], Y[i][1])
            vis.line(X=recall, Y=precision, win="PR Step:" + str(step),
                     name = str_ls[i],
                     update='append')


        #draw(Y[i][1], Y[i][0], "title")

        AUC.append(aucv)
    return AUC

#主程序部分

log = Logger(logname,level='debug')

#配置训练数据集&测试数据集
Train_Dataloader = DataLoader(Trainingdata(), batch_size = batch_size, shuffle=True, drop_last=True)

Test_CDT_Datalodaer=[]
for i in range(len(TACDTData)):
    Test_CDT_Datalodaer.append(DataLoader(Testdata(i), batch_size = batch_size, shuffle=False,drop_last=True))



model = Model().cuda()

LossF = torch.nn.BCELoss()

Optimizer = torch.optim.Adam(model.parameters(), lr=LR)

vis = visdom.Visdom(env=os.path.basename(__file__))
log.logger.debug('\n>>>>>New '+os.path.basename(__file__))
log.logger.debug('>>>>>Start Training')
log.logger.debug(get_parameter_number(model))


step=0
for e in range(epoch):

    model.train()

    TrainL = 0
    k = 0
    correct = 0
    for i, data in enumerate(Train_Dataloader):

        input=data['img']
        out=model(input) #内容屏蔽图像

        Train_loss = LossF(out, data['label'])

        Optimizer.zero_grad()
        Train_loss.backward()
        Optimizer.step()

        TrainL = TrainL + Train_loss.item()

        p=out.cpu().detach().numpy()
        gt=data['label'].cpu().detach().numpy()

        for c in range(len(p)):
            if p[c] > 0.5:
                pd = 1.
            else:
                pd = 0.
            if pd == gt[c]:
                correct = correct + 1


        k = k + batch_size

        if step%log_fre==0 :

            # vis.image(torch.clamp(input[0], 0, 1).cpu().detach().numpy()[batch_size - 1] * 255, win='input0',
            #           opts=dict(title=str(i) + ' input0'))
            # vis.image(torch.clamp(input[1], 0, 1).cpu().detach().numpy()[batch_size - 1] * 255, win='input1',
            #           opts=dict(title=str(i) + ' input1'))

            log.logger.debug(' e:' + str(e) +
                             ' ['+Tstring+'(vs.)'+Fstring+']Step:' + str(step) +
                             ' Loss:%.5f' % (TrainL / k) +
                             ' ACC(threshold=0.5):' + str(correct) + '/' + str(k) +
                             ' =%.5f' % (correct / k))

        if step % eval_frq == 0 and step>0:
            with torch.no_grad():

                # Test and cross domain Test
                log.logger.debug("Test&Valid...")
                model.eval()
                n = 0
                TestL = 0
                Y=[]
                for l in range(len(Test_CDT_Datalodaer)):

                    y_true = []  # 画pr曲线用
                    y_score = []

                    for _, TCDTData in enumerate(Test_CDT_Datalodaer[l]):
                        out = model(TCDTData['img'])

                        if l==0:#说明是test，应该记录loss以保存模型
                            Test_loss = LossF(out, TCDTData['label'])
                            TestL = TestL + Test_loss.item()
                            n = n + batch_size

                        p = out.cpu().detach().numpy()
                        gt = TCDTData['label'].cpu().detach().numpy()

                        for c in range(len(p)):
                            y_true.append(gt[c])
                            y_score.append(p[c])

                    Y.append([y_true,y_score])

                #绘制ROC PR曲线并返回AUC值
                AUC=ROC_AUC(vis, Y,step)

                s=''
                for ll in range(len(AUC)):
                    s=s+" \nAUC-" + str_ls[ll] + ":%.5f" % (AUC[ll])
                log.logger.debug('\n=== ' + str(e) +
                                 ' epoch-Testing loss:%.5f' % (TestL / n) +s)

                if step == eval_frq:
                    best = TestL

                if best > TestL:
                    best = TestL
                    LRChange = 0 #学习率计数清零
                    torch.save(model.state_dict(), 'WightModel/' + os.path.basename(__file__)+'best.pth')
                    log.logger.debug("==better model save!==")
                else:
                    LRChange = LRChange + 1
                    if LRChange > 3:  # 超过3代loss没有下降了 需要衰减学习率
                        LRChange = 0
                        LR = LR / 2

                        for param_group in Optimizer.param_groups:
                            param_group["lr"] = LR  # 更新学习率
                        log.logger.debug("\n===LR Change:" + str(LR) + "!===")

            model.train()

        step = step + 1

vis.save([os.path.basename(__file__)])