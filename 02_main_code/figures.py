import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report

import myconstant


def _ini_img_dir(ids):
    # 创建结果图文件夹
    if not os.path.exists(myconstant.SavedImages):
        os.makedirs(myconstant.SavedImages)
    if not os.path.exists(os.path.join(myconstant.SavedImages, ids)):
        os.makedirs(os.path.join(myconstant.SavedImages, ids))

    img_dir = os.path.join(myconstant.SavedImages, ids)
    return img_dir


def figure_all(txt_name, ids, X_train_raw, X_train, y_train, X_test, y_test, model_lassoCV, index, coef, model):
    feature_weight_figure(ids, index, coef)
    feature_correlation_figure(ids, X_train)
    lambda_selection_figure(ids, model_lassoCV)
    lambda_coefficient_figure(ids, model_lassoCV, X_train_raw, y_train)
    roc_curve_figure(ids, txt_name, model, X_test, y_test)


def feature_weight_figure(ids, index, coef):
    img_dir = _ini_img_dir(ids)

    # 特征权重图
    plt.figure(figsize=(10, 10), dpi=300)
    x_values = np.arange(len(index))
    y_values = coef[coef != 0]
    plt.bar(
        x_values, y_values,  # 横向bar使用：barh
        color='lightblue',  # 设置bar的颜色
        edgecolor='black',  # 设置bar边框颜色
        alpha=0.8,  # 设置不透明度
    )
    plt.xticks(
        x_values, index,
        rotation='45',  # 旋转xticks
        ha='right',  # xticks的水平对齐方式
        va='top',  # xticks的垂直对齐方式
        fontsize=12
    )
    plt.xlabel("feature")  # 横轴名称
    plt.ylabel("weight")  # 纵轴名称
    plt.title('Weight of features', fontsize=22)

    plt.savefig(f'{img_dir}/特征权重图.jpg', bbox_inches='tight', dpi=300)


def feature_correlation_figure(ids, X_train):
    img_dir = _ini_img_dir(ids)

    # 特征相关性热度图
    plt.figure(figsize=(10, 10), dpi=300)
    sns.heatmap(
        X_train.corr(),  # 计算特征间的相关性
        xticklabels=X_train.corr().columns,
        yticklabels=X_train.corr().columns,
        cmap='RdYlGn',
        center=0.5,
        annot=False
    )
    plt.title('Correlogram of features', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(f'{img_dir}/特征相关性热度图.jpg', bbox_inches='tight', dpi=300)


def lambda_selection_figure(ids, model_lassoCV):
    img_dir = _ini_img_dir(ids)

    # LASSO模型中Lambda选值图
    MSEs = model_lassoCV.mse_path_
    MSEs_mean = np.apply_along_axis(np.mean, 1, MSEs)
    MSEs_std = np.apply_along_axis(np.std, 1, MSEs)

    plt.figure(figsize=(10, 10), dpi=300)
    plt.errorbar(
        model_lassoCV.alphas_, MSEs_mean,  # x, y数据，一一对应，这里用的是alphas_，而不是alpha_
        yerr=MSEs_std,  # y误差范围
        fmt="o",  # 数据点标记
        ms=3,  # 数据点大小
        mfc="r",  # 数据点颜色
        mec="r",  # 数据点边缘颜色
        ecolor="lightblue",  # 误差棒颜色
        elinewidth=2,  # 误差棒线宽
        capsize=4,  # 误差棒边界线长度
        capthick=1  # 误差棒边界线厚度
    )
    plt.semilogx()  # 横坐标使用对数坐标
    plt.axvline(model_lassoCV.alpha_, color='black', ls="--")
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.title('Selection of lambda', fontsize=22)

    plt.savefig(f'{img_dir}/LASSO模型中Lambda选值图.jpg', bbox_inches='tight', dpi=300)


def lambda_coefficient_figure(ids, model_lassoCV, X_train_raw, y_train):
    img_dir = _ini_img_dir(ids)

    # 特征系数随Lambda变化曲线
    coef = model_lassoCV.path(X_train_raw, y_train, alphas=model_lassoCV.alphas_, max_iter=myconstant.MaxIter)[1].T

    plt.figure(figsize=(10, 10), dpi=300)
    plt.semilogx(model_lassoCV.alphas_, coef, '-')  # 横坐标使用对数坐标
    plt.axvline(model_lassoCV.alpha_, color='black', ls="--")
    plt.xlabel('Lambda')
    plt.ylabel('Coefficients')
    plt.title(f'Coefficient with lambda', fontsize=22)

    plt.savefig(f'{img_dir}/特征系数随Lambda变化曲线.jpg', bbox_inches='tight', dpi=300)


def roc_curve_figure(ids, txt_name, model, X_test, y_test):
    img_dir = _ini_img_dir(ids)

    y_test_probs = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_probs[:, 1], pos_label=1)
    plt.figure(figsize=(10, 10), dpi=300)
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")

    # 计算AUC
    y_test_pred = model.predict(X_test)
    auc_score = auc(fpr, tpr)
    myconstant.write_info(txt_name, 'ROC曲线的AUC为:', auc_score)

    #  精确度（Precision），敏感度（Sensitivity），特异度（Specificity）等输出
    #  将“1”类作为“阳性”时，“1”类的recall就是Sensitivity，“0”类的recall即为Specificity
    myconstant.write_info(txt_name, '将"1"类作为"阳性"时，"1"类的recall就是Sensitivity，"0"类的recall即为Specificity')
    myconstant.write_info(txt_name, '分类报告如下:', classification_report(y_test, y_test_pred))

    plt.title('ROC Curve', fontsize=22)
    plt.savefig(f'{img_dir}/ROC曲线.jpg', bbox_inches='tight', dpi=300)
