import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

import myconstant


def lasso(X_train, y_train, X_test):
    # LASSO特征筛选
    alphas = np.logspace(-4, 1, 100)  # alpha就是损失函数的lambda，从10的-4次方到10的1次方均匀地取50个值
    model_lassoCV = LassoCV(alphas=alphas, max_iter=myconstant.MaxIter, cv=10).fit(X_train, y_train)

    # coef_代表LASSO计算出来的每个特征的系数
    coef = pd.Series(model_lassoCV.coef_, index=X_train.columns)

    index = coef[coef != 0].index
    X_train_raw = X_train
    X_train = X_train[index]
    X_test = X_test[index]

    return X_train, X_test, index, coef, model_lassoCV
