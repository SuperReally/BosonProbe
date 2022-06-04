import os

sa="E:\Dataset\Datasets20220512"
sb='Dataset'
Trainnum=2500
Testnum=200
Validnum=600
S1='Train'
S2='Test'


FData= [sa+'/F-ProGAN-3K',
        sa+'/F-StyleGAN2-3K',
        sa+'/F-StyleGAN-3K']
TData= [ sa+'/T-AFHQ-3K/',
        sa+'/T-CelebaHQ-3K/',
        sa+'/T-FFHQ-3K/']

for l in range(len(FData)):
    # -----------------装入负例-------------------
    ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), FData[l])) if f.endswith('.png') or f.endswith('.jpg')]
    s = ''  # 给训练集添加
    for i in range(Trainnum):
        s = s + FData[l] + '/' + ImgPath[i] + ' F' + '\n'
    with open(sb + '/' + S1 + '.txt', 'a') as f:
        f.write(s)
        f.close()
    s = ''  # 给测试集集添加
    for i in range(Testnum):
        s = s + FData[l] + '/' + ImgPath[Trainnum + i] + ' F' + '\n'
    with open(sb + '/' + S2 + '.txt', 'a') as f:
        f.write(s)
        f.close()

    # -----------------装入正例-------------------
    ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), TData[l])) if f.endswith('.png') or f.endswith('.jpg')]
    s = ''  # 给训练集添加
    for i in range(Trainnum):
        s = s + TData[l] + '/' + ImgPath[i] + ' T' + '\n'
    with open(sb + '/' + S1 + '.txt', 'a') as f:
        f.write(s)
        f.close()
    s = ''  # 给测试集集添加
    for i in range(Testnum):
        s = s + TData[l] + '/' + ImgPath[Trainnum + i] + ' T' + '\n'
    with open(sb + '/' + S2 + '.txt', 'a') as f:
        f.write(s)
        f.close()



#-----------------RAW-------------------
#-------------跨域验证集1------------
S3='CD-CelebaHQvsStyleGAN3'
path=sa+'/V-F-StyleGAN3-600'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/T-CelebaHQ-3K'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[2400+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()

#-------------跨域验证集2------------
S3='CD-FFHQvsStyleGAN3'
path=sa+'/V-F-StyleGAN3-600'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s='' #给训练集添加
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/T-FFHQ-3K'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s='' #给训练集添加
for i in range(Validnum):
    s= s +path+'/'+ImgPath[2400+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()

#-------------跨域验证集3------------
S3='CD-AFHQvsStyleGAN3'
path=sa+'/V-F-StyleGAN3-600'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/T-AFHQ-3K'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[2400+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()

#-------------跨域验证集4------------
S3='CD-MetafacesvsStyleGAN3'
path=sa+'/V-F-StyleGAN3-600'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/V-T-Metafaces-600'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s='' #给训练集添加
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()

#-------------跨域验证集5------------
S3='CD-LSUNvsStyleGAN3'
path=sa+'/V-F-StyleGAN3-600'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/V-T-LSUN-600'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s='' #给训练集添加
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()


#-----------------LQ-------------------
#-------------跨域验证集1------------
S3='CD-CelebaHQvsStyleGAN3-LQ'
path=sa+'/V-F-StyleGAN3-600-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/T-CelebaHQ-3K-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[2400+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()

#-------------跨域验证集2------------
S3='CD-FFHQvsStyleGAN3-LQ'
path=sa+'/V-F-StyleGAN3-600-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s='' #给训练集添加
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/T-FFHQ-3K-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s='' #给训练集添加
for i in range(Validnum):
    s= s +path+'/'+ImgPath[2400+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()

#-------------跨域验证集3------------
S3='CD-AFHQvsStyleGAN3-LQ'
path=sa+'/V-F-StyleGAN3-600-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/T-AFHQ-3K-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[2400+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()

#-------------跨域验证集4------------
S3='CD-MetafacesvsStyleGAN3-LQ'
path=sa+'/V-F-StyleGAN3-600-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/V-T-Metafaces-600-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s='' #给训练集添加
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()

#-------------跨域验证集5------------
S3='CD-LSUNvsStyleGAN3-LQ'
path=sa+'/V-F-StyleGAN3-600-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/V-T-LSUN-600-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s='' #给训练集添加
for i in range(Validnum):
    s= s +path+'/'+ImgPath[i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()









#-------------global test------------
S3='CD-Global'

#-------------------------------
#300张raw stylegan3
path=sa+'/V-F-StyleGAN3-600'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(300):
    s= s +path+'/'+ImgPath[i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
#300张lq stylegan3
path=sa+'/V-F-StyleGAN3-600-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(300):
    s= s +path+'/'+ImgPath[300+i] + ' F' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()

#-------------------------------
#60张raw
path=sa+'/T-CelebaHQ-3K'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(60):
    s= s +path+'/'+ImgPath[2400+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
#60张lq
path=sa+'/T-CelebaHQ-3K-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(60):
    s= s +path+'/'+ImgPath[2500+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()

#-------------------------------
path=sa+'/T-FFHQ-3K'
#60张raw
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(60):
    s= s +path+'/'+ImgPath[2400+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
#60张lq
path=sa+'/T-FFHQ-3K-LQ'
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(60):
    s= s +path+'/'+ImgPath[2500+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()


#-------------------------------
path=sa+'/T-AFHQ-3K'
#60张raw
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(60):
    s= s +path+'/'+ImgPath[2400+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/T-AFHQ-3K-LQ'
#60张lq
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(60):
    s= s +path+'/'+ImgPath[2500+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()

#-------------------------------
path=sa+'/V-T-Metafaces-600'
#60张raw
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(60):
    s= s +path+'/'+ImgPath[i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/V-T-Metafaces-600-LQ'
#60张lq
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(60):
    s= s +path+'/'+ImgPath[100+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()


#-------------------------------
path=sa+'/V-T-LSUN-600'
#60张raw
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(60):
    s= s +path+'/'+ImgPath[i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
path=sa+'/V-T-LSUN-600-LQ'
#60张lq
ImgPath = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.png') or f.endswith('.jpg')]
s=''
for i in range(60):
    s= s +path+'/'+ImgPath[100+i] + ' T' + '\n'
with open(sb+'/'+S3+'.txt', 'a') as f:
    f.write(s)
    f.close()
