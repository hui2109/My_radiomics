import os
import shutil


def select_jpg_json(images_path: str, prefix: str, des_path: str, selected_images: list):
    file_types = ['.jpg', '.json']

    for types in file_types:
        images_name = []
        for i in selected_images:
            images_name.append(prefix + i + types)
        # print(images_name)

        for image in os.listdir(images_path):
            if image in images_name:
                des = os.path.join(des_path, image)
                src = os.path.join(images_path, image)
                shutil.copy(src, des)
            else:
                print(image)


if __name__ == '__main__':
    with open('../00_资源库/source_data/poor_post.txt', 'r', 1, 'utf-8') as f:
        s_list = f.read().split(' ')
    print(s_list, len(set(s_list)))  # 237 534
    select_jpg_json('../00_资源库/source_data/poor_post', 'poor_post_', '../00_资源库/source_data/poor_post_s', s_list)
