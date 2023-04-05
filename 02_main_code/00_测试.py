# import radiomics
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# import glob
#
# # 查看其中一个图片
# mask_1=Image.open(r'D:\桌面\img\img_1.png')
# s_1=np.array(mask_1.convert('L'))
# print(s_1.shape)
#
# mask_2=Image.open(r'D:\桌面\img\img_2.png')
# s_2=np.array(mask_2.convert('L'))
# print(s_2.shape)
#
# # 将list转为np.array
# allmask_array=[]
# allmask_array.append(s_1)
# allmask_array.append(s_2)
# allmask_array=np.array(allmask_array)
# print(allmask_array.shape)
#
# # 通过sitk的包读取array，保存为nii
# out_mask = sitk.GetImageFromArray(np.array(allmask_array))
# sitk.WriteImage(out_mask,r'D:\out_mask.nii.gz')
#
# # itk_img=sitk.ReadImage(r'D:\out_mask.nii.gz')
# # img = sitk.GetArrayFromImage(itk_img)
# # print(img.shape)
# # plt.imshow(img[0],cmap='gray')  # 展示其中一张有mask的图片
# # plt.show()
#
# # from PIL import Image
# # import numpy as np
# # img = Image.open(r"D:\肝包虫图片\包虫勾画工作\专家共识包虫图片\4_陈娟\2016年超声图片\41036\41036-1212\2016_41036_1.jpg")
# img.save(r"D:\桌面\945.png")
#
# # img_1=r'D:\肝包虫图片\包虫勾画工作\专家共识包虫图片\4_陈娟\2016年超声图片\41036\41036-1212\2016_41036_1.jpg'
# # img_2=r'D:\肝包虫图片\包虫勾画工作\专家共识包虫图片\1_任叶蕾\2014年超声图片\158\2014_158_1.bmp'
# #
# # image_1=Image.open(img_1)
# # print(image_1.size)
# # i_1=np.array(image_1.convert('L'))
# # print(i_1.shape)
# #
# # a=Image.fromarray(i_1, 'L')
# # a.save(r'D:/桌面/1.jpg')
#
# # image_2=Image.open(img_2)
# # print(image_2.size)
# # image_3=image_2.resize((800, 600))
# # image_3.save(r'D:/桌面/3.jpg')
# # i_2=np.array(image_3.convert('L'))
# # print(i_2.shape)

# coding:utf8
import os
import cv2 as cv
import numpy as np
# coding:utf8
import os

import cv2 as cv
import numpy as np

# 二值化后图像保存路径
des_path = r'E:\1Test\5data\3004325563\\'


def file_rename(path):
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    # print(filelist)
    for files in sorted(filelist):
        if files.lower().endswith('_mask.png'):
            image = cv.imread(path + files)
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # 把输入图像灰度化
            h, w = gray.shape[:2]
            m = np.reshape(gray, [1, w * h])
            mean = m.sum() / (w * h)
            print("mean:", mean)
            ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
            cv.imwrite(des_path + files, binary)


if __name__ == '__main__':
    path = r'E:\1Test\4label\3004325563\\'  # 全部文件的路径
    file_rename(path)
