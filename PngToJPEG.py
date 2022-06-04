# -- coding: utf-8 --
import os

from PIL import Image

sa="E:\Dataset\Datasets20220512"
path=sa+'/F-StyleGAN2-3K'
path1=sa+'/F-StyleGAN2-3K-ToJPEG'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
for i in range(len(ImgPath)):
    img = Image.open(path+"/"+ImgPath[i])
    #img = img.convert('RGB')
    img.save(path1+"/"+ImgPath[i][:-4]+".jpg", quality=70)