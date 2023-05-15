import pandas as pd
from sklearn.model_selection import train_test_split


def data_preprocess(path, start_column_num=22):
    # 数据预处理
    data = pd.read_excel(path).iloc[:, start_column_num:]

    # 数据集划分
    # test_size 7 3 分, random_state 每次运行都是一个稳定的结果
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=15)
    data_train_a = data_train[data_train['label'] == 0]
    data_train_b = data_train[data_train['label'] == 1]
    data_test_a = data_test[data_test['label'] == 0]
    data_test_b = data_test[data_test['label'] == 1]

    return data, data_train_a, data_train_b, data_test_a, data_test_b
