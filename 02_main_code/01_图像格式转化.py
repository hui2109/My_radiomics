import os
from collections import OrderedDict

import SimpleITK as sitk
import numpy as np
from PIL import Image


class ImagesToNii:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_names = os.listdir(self.image_dir)
        self.widths = []
        self.heights = []

    def getImageInfo(self):
        self.image_info = OrderedDict()
        widths = []
        heights = []
        for image_name in self.image_names:
            image_path = os.path.join(self.image_dir, image_name)
            image_obj = Image.open(image_path)

            widths.append(image_obj.size[0])
            heights.append(image_obj.size[1])
            self.image_info[image_name] = {'width': image_obj.size[0],
                                           'height': image_obj.size[1],
                                           'format': image_obj.format,
                                           'image_self': image_obj}
        return widths, heights

    def getImageSize(self):
        widths, heights = self.getImageInfo()
        better_width = max(widths)
        better_height = max(heights)
        return better_width, better_height

    def imagesToNii(self):
        size = self.getImageSize()
        all_image_array = []
        for items in self.image_info.values():  # item是字典
            image_obj = items['image_self']
            new_image_obj = image_obj.resize(size)
            new_image_array = np.array(new_image_obj.convert('L'))
            all_image_array.append(new_image_array)
        all_image_array = np.array(all_image_array)
        out_images = sitk.GetImageFromArray(np.array(all_image_array))
        sitk.WriteImage(out_images, './label_CE.nii.gz')


if __name__ == '__main__':
    image_dir = './测试图像'
    idn = ImagesToNii(image_dir)
    idn.imagesToNii()
