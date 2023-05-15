from sklearn.svm import SVC


def svm(X_train, y_train, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=20):
    # 支持向量机分类器
    model = SVC(kernel=kernel,  # 常用的核函数有：
                # 线性核函数（linear），多项式核函数（poly），径向基核函数（rbf），sigmoid核函数
                gamma=gamma,  # gamma定义了单个样本的影响范围，gamma越大，支持向量越多，也可以写auto、浮点数
                probability=probability,    # 后面可以看分到哪一类的概率
                class_weight=class_weight,  # 类别的权重, 目的是让0类和1类的数据平衡, 默认是None
                random_state=random_state,  # 每次运行都是一个稳定的结果, 默认是None
                )
    model.fit(X_train, y_train)

    return model
