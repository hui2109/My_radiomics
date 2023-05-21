import os
import random
from shutil import copy

if not os.path.exists('../00_资源库/dataset_voc'):
    os.makedirs('../00_资源库/dataset_voc/JPEGImages')
    os.makedirs('../00_资源库/dataset_voc/SegmentationClassPNG')


def makeup_data(label='good', num=125):
    prefix = f'../00_资源库/source_data/dataset_voc_{label}'
    keys = set()
    selected_images = {}

    jpg_images = os.listdir(f'{prefix}/JPEGImages')
    jpg_images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    png_images = os.listdir(f'{prefix}/SegmentationClassPNG')
    png_images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    images_map = dict(zip(jpg_images, png_images))
    # 测试是否正确
    # for key, value in images_map.items():
    #     if key.split('.')[0] != value.split('.')[0]:
    #         print('error')

    # 随机选择图像
    if num <= len(images_map):
        # 可能选到重复的值
        while True:
            key = random.choice(list(images_map.keys()))
            keys.add(key)
            if len(keys) == num:
                break
        for key in keys:
            selected_images[key] = images_map[key]
    else:
        print('输入的num值太大')

    return selected_images


def move_images(selected_images: dict, label='good'):
    for key in selected_images:
        des_path = '../00_资源库/dataset_voc/JPEGImages/' + key
        sor_path = f'../00_资源库/source_data/dataset_voc_{label}/JPEGImages/' + key
        copy(sor_path, des_path)

        des_path = '../00_资源库/dataset_voc/SegmentationClassPNG/' + selected_images[key]
        sor_path = f'../00_资源库/source_data/dataset_voc_{label}/SegmentationClassPNG/' + selected_images[key]
        copy(sor_path, des_path)


if __name__ == '__main__':
    selected_images = makeup_data('good', num=237)
    move_images(selected_images, 'good')

    selected_images = makeup_data('poor', num=535)
    move_images(selected_images, 'poor')
