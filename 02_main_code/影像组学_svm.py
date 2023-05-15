import figures as fg
import myconstant
from batchextract import batch_extract
from datapreprocess import data_preprocess
from imgtonrrd import image_to_nrrd
from lasso import lasso
from svm import svm
from t_test import t_test


def svm_path():
    # 初始化记录本
    txt_name, ids = myconstant.init_info('svm')
    # 将图像转换为nrrd格式
    image_to_nrrd(38)
    # 批量提取特征，得到特征表
    result_path = batch_extract('../00_资源库/feature_results/results_1684150306.xlsx')
    # 数据预处理及数据集划分
    data, data_train_a, data_train_b, data_test_a, data_test_b = data_preprocess(result_path)
    print('训练集，label为0的形状', data_train_a.shape, '\n',
          '训练集，label为1的形状', data_train_b.shape, '\n',
          '测试集，label为0的形状', data_test_a.shape, '\n',
          '测试集，label为1的形状', data_test_b.shape)

    # 进行t检验，分别得到：
    # 训练集的特征数据、训练集的标签数据、测试集的特征数据、测试集的标签数据、筛选出的特征名
    X_train, y_train, X_test, y_test, index = t_test(data, data_train_a, data_train_b, data_test_a, data_test_b)
    myconstant.write_info(txt_name, 't检验后, 筛选出', len(index), '个特征')
    myconstant.write_info(txt_name, '这些特征分别是:', index)
    # X_train_raw用于画图
    X_train_raw = X_train.copy()

    # 进行lasso回归，分别得到：
    # 训练集的特征数据、测试集的特征数据、筛选出的特征名、每个特征的对应系数、lasso模型
    X_train, X_test, index, coef, model_lassoCV = lasso(X_train, y_train, X_test)
    # alpha_代表选出来的最优alpha值
    myconstant.write_info(txt_name, '最优alpha值为:', model_lassoCV.alpha_)
    myconstant.write_info(txt_name, '%s %d' % ('Lasso选择出的系数不为0的特征的个数:', sum(coef != 0)))
    myconstant.write_info(txt_name, '这些特征及其系数分别是:', dict(zip(index, coef[coef != 0])))

    # 支持向量机分类器
    model = svm(X_train, y_train)
    myconstant.write_info(txt_name, '训练集上的平均准确率为:', model.score(X_train, y_train))
    myconstant.write_info(txt_name, '测试集上的平均准确率为:', model.score(X_test, y_test))
    myconstant.write_info(txt_name, '测试集的正确结果为:', list(y_test))
    myconstant.write_info(txt_name, '对测试集进行结果预测的展示:', model.predict(X_test))
    myconstant.write_info(txt_name, '测试集中每种预测结果的概率展示:', model.predict_proba(X_test))
    myconstant.write_info(txt_name, '在拟合过程中用到了多少特征:', model.n_features_in_)
    myconstant.write_info(txt_name, '展示支持向量机模型的各个参数:', model.get_params())

    # 绘图
    fg.figure_all(txt_name, ids, X_train_raw, X_train, y_train, X_test, y_test, model_lassoCV, index, coef, model)


svm_path()
