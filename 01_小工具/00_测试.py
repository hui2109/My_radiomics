import radiomics
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob

# 查看其中一个图片
mask_1=Image.open(r'D:\桌面\img\img_1.png')
s_1=np.array(mask_1.convert('L'))
print(s_1.shape)

mask_2=Image.open(r'D:\桌面\img\img_2.png')
s_2=np.array(mask_2.convert('L'))
print(s_2.shape)

# 将list转为np.array
allmask_array=[]
allmask_array.append(s_1)
allmask_array.append(s_2)
allmask_array=np.array(allmask_array)
print(allmask_array.shape)

# 通过sitk的包读取array，保存为nii
out_mask = sitk.GetImageFromArray(np.array(allmask_array))
sitk.WriteImage(out_mask,r'D:\out_mask.nii.gz')

# itk_img=sitk.ReadImage(r'D:\out_mask.nii.gz')
# img = sitk.GetArrayFromImage(itk_img)
# print(img.shape)
# plt.imshow(img[0],cmap='gray')  # 展示其中一张有mask的图片
# plt.show()
