from sklearn.linear_model import LogisticRegressionCV


def logistic(X_train, y_train, cv=10, class_weight='balanced', random_state=20):
    # 支持向量机分类器
    model = LogisticRegressionCV(cv=cv, class_weight=class_weight, random_state=random_state)
    model.fit(X_train, y_train)

    return model
