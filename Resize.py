import os
import cv2

sa="E:\Dataset\Datasets20220512"

RawData= [sa+'/F-ProGAN-3K/',
        sa+'/F-StyleGAN2-3K/',
        sa+'/F-StyleGAN-3K/',
        sa+'/T-AFHQ-3K/',
        sa+'/T-CelebaHQ-3K/',
        sa+'/T-FFHQ-3K/',
        sa+'/V-F-StyleGAN3-600/',
        sa+'/V-T-LSUN-600/',
        sa+'/V-T-Metafaces-600/']

LQData= [sa+'/F-ProGAN-3K-LQ/',
        sa+'/F-StyleGAN2-3K-LQ/',
        sa+'/F-StyleGAN-3K-LQ/',
        sa+'/T-AFHQ-3K-LQ/',
        sa+'/T-CelebaHQ-3K-LQ/',
        sa+'/T-FFHQ-3K-LQ/',
        sa+'/V-F-StyleGAN3-600-LQ/',
        sa+'/V-T-LSUN-600-LQ/',
        sa+'/V-T-Metafaces-600-LQ/']

for i in range(len(RawData)):

    print("current task:"+RawData[i])
    imgtT = [f for f in os.listdir(os.path.join(os.getcwd(), RawData[i])) if f.endswith('.jpg') or f.endswith('.png')]

    for ii in range(len(imgtT)):
        img = cv2.imread(os.path.join(os.getcwd(), RawData[i]) + imgtT[ii])
        new_h=int(img.shape[0]*0.8)
        new_w = int(img.shape[1] * 0.8)
        if new_h<256 or new_w<256:
            print('图像太小，不做处理')
            cv2.imwrite(os.path.join(os.getcwd(), LQData[i] + imgtT[ii]), img)
        else:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(os.getcwd(), LQData[i] + imgtT[ii]), img)



