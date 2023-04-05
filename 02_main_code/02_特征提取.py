import SimpleITK as sitk
import numpy as np
import pandas as pd
from PIL import Image
from radiomics import featureextractor


def extract_features(img, mask):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])
    extractor.enableFeatureClassByName('glcm')
    extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})

    return extractor.execute(img, mask)


# reader = sitk.ImageFileReader()
# reader.SetImageIO('PNGImageIO')
# reader.SetFileName(r'D:\data\img.png')
# image = reader.Execute()
# print(image.GetDimension())
# print(image.GetSize())

# image = sitk.ReadImage(r'D:\data\img.png')
# print(image.GetDimension())
# print(image.GetSize())
# nda=sitk.GetArrayFromImage(image)
# print(nda.shape)
# new_nda=np.array([nda])
# print(new_nda.shape)
# img = sitk.GetImageFromArray(nda, isVector=True)
# print(img.GetDimension())
# print(img.GetSize())

img = Image.open(r'D:\data\2019_2655_6.jpg')
grey_img = img.convert('L')
grey_img_array = np.array(grey_img)
grey_img_3d_array = np.array([grey_img_array])
img = sitk.GetImageFromArray(grey_img_3d_array)
# size = img.GetSize() # order: x, y, z
# origin = img.GetOrigin() # order: x, y, z
# spacing = img.GetSpacing() # order:x, y, z
# direction = img.GetDirection() # order: x, y, z
# print(size,origin,spacing,direction)
# print(img.GetDimension())
# print(img.GetSize())
# print(type(img))

label = Image.open(r'D:\data\2019_2655_6.png')
grey_label = label.convert('L')
grey_label_array = np.array(grey_label)
grey_label_3d_array = np.array([grey_label_array])
label = sitk.GetImageFromArray(grey_label_3d_array)
size = label.GetSize()  # order: x, y, z
origin = label.GetOrigin()  # order: x, y, z
spacing = label.GetSpacing()  # order:x, y, z
direction = label.GetDirection()  # order: x, y, z
# print(size,origin,spacing,direction)

maskArr = sitk.GetArrayFromImage(label)


# 获取ROI的值
zero_mask = (maskArr != 0)
roi_value_set = set(maskArr[zero_mask])

# 迭代勾画的每一个roi
df = pd.DataFrame()

for roi_value in roi_value_set:
    maskArr_new = maskArr.copy()
    maskArr_new[maskArr != roi_value] = 0
    maskArr_new[maskArr == roi_value] = 1
    mask_new = sitk.GetImageFromArray(maskArr_new)
    mask_new.SetDirection(direction)
    mask_new.SetSpacing(spacing)
    mask_new.SetOrigin(origin)
    featureVector = extract_features(img=img, mask=mask_new)
    featureVector['label'] = roi_value
    df_new = pd.DataFrame.from_dict(featureVector.values()).T
    df_new.columns = featureVector.keys()
    df = pd.concat([df, df_new])
# size = mask_new.GetSize() # order: x, y, z
# origin = mask_new.GetOrigin() # order: x, y, z
# spacing = mask_new.GetSpacing() # order:x, y, z
# direction = mask_new.GetDirection() # order: x, y, z
# print(size,origin,spacing,direction)


# print(maskArr_new)
# print(maskArr.shape)
# sitk.WriteImage(label,'./label.nii')
# print(label.GetDimension())
# print(label.GetSize())
# print(type(label))
# df.to_excel(os.path.join('results.xlsx'), )

with pd.ExcelWriter('results.xlsx') as writer:
    df.to_excel(writer, index=False)
