import os.path
import time

DatasetVoc = '../00_资源库/dataset_voc'
JPEGImages = '../00_资源库/dataset_voc/JPEGImages'
SegmentationClassPNG = '../00_资源库/dataset_voc/SegmentationClassPNG'
NrrdFiles = '../00_资源库/nrrd_files'
PKLFile = '../00_资源库/pickle_file.pkl'
FeatureResults = '../00_资源库/feature_results'
InfoTexts = '../00_资源库/info_texts'
SavedImages = '../00_资源库/saved_images'
MaxIter = 1000000


def init_info(method):
    if not os.path.exists(InfoTexts):
        os.makedirs(InfoTexts)
    ids = f'info_{method}_{int(time.time())}'
    txt_name = f'{InfoTexts}/{ids}.txt'
    with open(txt_name, 'a', 1, 'utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' * 4 + f'{method}')
        f.write('\n')
    return txt_name, ids


def write_info(path, *msg):
    with open(path, 'a', 1, 'utf-8') as f:
        for i in msg:
            f.write(str(i))
        f.write('\n')
