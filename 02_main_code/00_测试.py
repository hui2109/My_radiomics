import os

path = r'D:\exe_code\FAE\fans_data\ceus_post'
des_path = r'D:\exe_code\FAE\fans_data\b_pre'

for p in os.listdir(path):
    if os.path.isdir(os.path.join(path, p)):
        os.makedirs(os.path.join(des_path, p))
