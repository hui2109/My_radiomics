import pandas as pd
from scipy.stats import levene, ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def t_test(data, data_train_a, data_train_b, data_test_a, data_test_b):
    # t检验特征筛选
    index = []
    for colName in data.columns[:]:
        if levene(data_train_a[colName], data_train_b[colName])[1] > 0.05:
            if ttest_ind(data_train_a[colName], data_train_b[colName])[1] < 0.05:
                index.append(colName)
        else:
            if ttest_ind(data_train_a[colName], data_train_b[colName], equal_var=False)[1] < 0.05:
                index.append(colName)

    # t检验后训练集数据整理
    data_train_a = data_train_a[index]
    data_train_b = data_train_b[index]
    data_train = pd.concat([data_train_a, data_train_b])
    data_train = shuffle(data_train)
    data_train.index = range(len(data_train))  # 打乱后重新赋值索引
    X_train = data_train[data_train.columns[:-1]]

    # 注意下面两行在训练集与测试集上的区别; 因为是同一组数据, 标准化的方法要一致
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_train.columns = index[:-1]
    y_train = data_train['label']

    # t检验后测试集数据整理
    data_test_a = data_test_a[index]
    data_test_b = data_test_b[index]
    data_test = pd.concat([data_test_a, data_test_b])
    data_test = shuffle(data_test)
    data_test.index = range(len(data_test))  # 打乱后重新赋值索引
    X_test = data_test[data_test.columns[:-1]]
    X_test = scaler.transform(X_test)  # 这里注意, 使用的是同一把尺子
    X_test = pd.DataFrame(X_test)
    X_test.columns = index[:-1]
    y_test = data_test['label']

    return X_train, y_train, X_test, y_test, index
