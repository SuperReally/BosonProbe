import os
import cv2



path= 'E:\Dataset\Datasets20220512\V-T-LSUN-600\\'

imgtT = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.jpg') or f.endswith('.png')]

for i in range(len(imgtT)):
    img = cv2.imread(os.path.join(os.getcwd(), path) + imgtT[i])
    # print(img.shape)
    if img.shape[0]<256 or img.shape[1]<256:
        print(os.path.join(os.getcwd(), path) + imgtT[i])
