import os
import shutil


def select_jpg_json(images_path: str, prefix: str, des_path: str, selected_images: list):
    file_types = ['.jpg', '.json']
    # file_types = ['.json']

    if not os.path.exists(des_path):
        os.makedirs(des_path)

    for types in file_types:
        # a = []
        images_name = []
        # print(len(selected_images))
        for i in selected_images:
            images_name.append(prefix + i + types)
        # print(len(images_name))

        for image in os.listdir(images_path):
            # print(len(images_name))
            if image in images_name:
                # a.append(image)
                des = os.path.join(des_path, image)
                src = os.path.join(images_path, image)
                shutil.copy(src, des)
                # pass
            else:
                # if image.replace(prefix, '').replace(types, '') not in selected_images:
                #     print(image)
                pass
        # print(a, len(a))
        # b = []
        # for kl in a:
        #     m = kl.replace(prefix, '').replace(types, '')
        #     b.append(m)
        # print(len(selected_images), len(b))
        # for op in selected_images:
        #     if op not in b:
        #         print(op)


if __name__ == '__main__':
    with open('../00_resources/source_data/b_post/good_post.txt', 'r', 1, 'utf-8') as f:
        s_list = f.read().split(' ')
    # print(s_list, len(s_list), len(set(s_list)))  # good_post: 502, poor_post: 1380
    # for i in s_list:
    #     if int(i) > 3440:
    #         print('no', i)
    # c = [n for n in s_list if s_list.count(n) > 1]
    # print(c)
    select_jpg_json('../00_resources/source_data/b_post/good_post', 'good_post_', '../00_resources/source_data/b_post/good_post_s', s_list)
