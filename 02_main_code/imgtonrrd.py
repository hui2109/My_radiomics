import os
import pickle

import SimpleITK as sitk
import numpy as np
from PIL import Image

import myconstant


def image_to_nrrd(negative=38, use_pickle=True):
    """
    :param negative: 红色值: 38; 绿色值: 75
    :param use_pickle: 是否使用pickle文件
    :return: 无返回值
    """

    # 有pickle文件就不用再提取了
    if os.path.exists(myconstant.PKLFile) and use_pickle:
        return None

    if not os.path.exists(myconstant.DatasetVoc):
        print('资源库文件夹下没有dataset_voc文件夹')
    if not (os.path.exists(myconstant.JPEGImages) and os.path.isdir(myconstant.SegmentationClassPNG)):
        print('dataset_voc文件夹下没有JPEGImages文件夹和SegmentationClassPNG文件夹')

    finding_list = []
    for i in os.listdir(myconstant.JPEGImages):
        for j in os.listdir(myconstant.SegmentationClassPNG):
            if i.split('.')[0] == j.split('.')[0]:
                img = Image.open(os.path.join(myconstant.JPEGImages, i))
                grey_img = img.convert('L')
                grey_img_array = np.array(grey_img)
                grey_img_3d_array = np.array([grey_img_array])
                img = sitk.GetImageFromArray(grey_img_3d_array)

                label = Image.open(os.path.join(myconstant.SegmentationClassPNG, j))
                grey_label = label.convert('L')
                grey_label_array = np.array(grey_label)
                grey_label_3d_array = np.array([grey_label_array])
                label = sitk.GetImageFromArray(grey_label_3d_array)

                origin = label.GetOrigin()
                spacing = label.GetSpacing()
                direction = label.GetDirection()

                # 获取ROI的值
                zero_mask = (grey_label_3d_array != 0)
                roi_value_set = set(grey_label_3d_array[zero_mask])

                # 迭代勾画的每一个roi
                for roi_value in roi_value_set:
                    group_dict = {}
                    maskArr = grey_label_3d_array.copy()
                    maskArr[grey_label_3d_array != roi_value] = 0
                    maskArr[grey_label_3d_array == roi_value] = 1

                    roi_value = 0 if roi_value == negative else 1

                    mask = sitk.GetImageFromArray(maskArr)
                    mask.SetDirection(direction)
                    mask.SetSpacing(spacing)
                    mask.SetOrigin(origin)

                    # 将数组写入文件
                    if not os.path.exists(myconstant.NrrdFiles):
                        os.makedirs(myconstant.NrrdFiles)

                    sitk.WriteImage(img, f'{myconstant.NrrdFiles}/{i.split(".")[0]}_img.nrrd')
                    sitk.WriteImage(mask, f'{myconstant.NrrdFiles}/{j.split(".")[0]}_mask_{roi_value}.nrrd')

                    # 制作成组表,方便查找
                    group_dict['img'] = f'{myconstant.NrrdFiles}/{i.split(".")[0]}_img.nrrd'
                    group_dict['mask'] = f'{myconstant.NrrdFiles}/{j.split(".")[0]}_mask_{roi_value}.nrrd'
                    group_dict['roi_value'] = str(roi_value)
                    finding_list.append(group_dict)

    with open(myconstant.PKLFile, 'wb') as f:
        pickle.dump(finding_list, f)
