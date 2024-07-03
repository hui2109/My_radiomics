import myconstant
from batchextract import batch_extract
from datapreprocess import data_preprocess
from imgtonrrd import image_to_nrrd


def get_features():
    # 初始化记录本
    txt_name, ids = myconstant.init_info('get_features')
    # 将图像转换为nrrd格式
    image_to_nrrd(75, use_pickle=False)
    # 批量提取特征，得到特征表
    result_path = batch_extract()
    # 数据预处理及数据集划分
    data, data_train_a, data_train_b, data_test_a, data_test_b = data_preprocess(result_path)
    print('训练集，label为0的形状', data_train_a.shape, '\n',
          '训练集，label为1的形状', data_train_b.shape, '\n',
          '测试集，label为0的形状', data_test_a.shape, '\n',
          '测试集，label为1的形状', data_test_b.shape)


get_features()
