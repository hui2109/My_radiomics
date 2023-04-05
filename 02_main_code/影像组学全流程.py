import os
import pickle
import time

import SimpleITK as sitk
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from radiomics import featureextractor
from scipy.stats import ttest_ind, levene
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle


def _extract_features(img, mask):
    extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
    # extractor.disableAllFeatures()
    # extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])
    # extractor.enableFeatureClassByName('glcm')
    extractor.enableAllFeatures()
    extractor.enableImageTypes(
        Original={},
        LoG={},
        Wavelet={},
        Square={},
        SquareRoot={},
        Logarithm={},
        Exponential={},
        Gradient={},
        LBP2D={}
    )

    return extractor.execute(img, mask)


class RadiomicsAnalysis:
    def __init__(self):
        pass

    def svm_route(self):
        self.batch_extract()
        self.data_preprocess()
        self.t_test()
        self.lasso()
        self.svm()
        self.feature_weight_figure()
        self.feature_correlation_figure()
        self.lambda_selection_figure()
        self.lambda_coefficient_figure()
        self.roc_curve_figure()

    def rf_route(self):
        self.batch_extract()
        self.data_preprocess()
        self.t_test()
        self.lasso()
        self.random_forest()
        self.feature_weight_figure()
        self.feature_correlation_figure()
        self.lambda_selection_figure()
        self.lambda_coefficient_figure()
        self.roc_curve_figure()

    def pca_route(self):
        self.batch_extract()
        self.data_preprocess()
        self.t_test()
        self.lasso()
        self.pca()
        self.random_forest()
        self.feature_weight_figure()
        self.feature_correlation_figure()
        self.lambda_selection_figure()
        self.lambda_coefficient_figure()
        self.roc_curve_figure()

    def _image_to_nrrd(self):
        if not os.path.exists('dataset_voc/'):
            print('当前文件夹下没有dataset_voc文件夹')
        if not (os.path.exists('dataset_voc/JPEGImages/') and os.path.isdir('dataset_voc/SegmentationClassPNG/')):
            print('dataset_voc文件夹下没有JPEGImages文件夹和SegmentationClassPNG文件夹')

        finding_list = []
        for i in os.listdir('dataset_voc/JPEGImages'):
            for j in os.listdir('dataset_voc/SegmentationClassPNG'):
                if i.split('.')[0] == j.split('.')[0]:
                    img = Image.open(os.path.join('dataset_voc/JPEGImages', i))
                    grey_img = img.convert('L')
                    grey_img_array = np.array(grey_img)
                    grey_img_3d_array = np.array([grey_img_array])
                    img = sitk.GetImageFromArray(grey_img_3d_array)

                    label = Image.open(os.path.join('dataset_voc/SegmentationClassPNG', j))
                    grey_label = label.convert('L')
                    grey_label_array = np.array(grey_label)
                    grey_label_3d_array = np.array([grey_label_array])
                    label = sitk.GetImageFromArray(grey_label_3d_array)

                    origin = label.GetOrigin()
                    spacing = label.GetSpacing()
                    direction = label.GetDirection()

                    # 获取ROI的值
                    zero_mask = (grey_label_3d_array != 0)
                    roi_value_set = set(grey_label_3d_array[zero_mask])

                    # 迭代勾画的每一个roi
                    for roi_value in roi_value_set:
                        group_dict = {}
                        maskArr = grey_label_3d_array.copy()
                        maskArr[grey_label_3d_array != roi_value] = 0
                        maskArr[grey_label_3d_array == roi_value] = 1

                        roi_value = 0 if roi_value == 38 else 1

                        mask = sitk.GetImageFromArray(maskArr)
                        mask.SetDirection(direction)
                        mask.SetSpacing(spacing)
                        mask.SetOrigin(origin)

                        # 将数组写入文件
                        if not os.path.exists('./nrrd_file/'):
                            os.makedirs('./nrrd_file/')
                        sitk.WriteImage(img, f'./nrrd_file/{i.split(".")[0]}_img.nrrd')
                        sitk.WriteImage(mask, f'./nrrd_file/{j.split(".")[0]}_mask_{roi_value}.nrrd')

                        # 制作成组表,方便查找
                        group_dict['img'] = f'./nrrd_file/{i.split(".")[0]}_img.nrrd'
                        group_dict['mask'] = f'./nrrd_file/{j.split(".")[0]}_mask_{roi_value}.nrrd'
                        group_dict['roi_value'] = str(roi_value)
                        finding_list.append(group_dict)

        with open('./nrrd_file/pickle_file.txt', 'wb') as f:
            pickle.dump(finding_list, f)

    def batch_extract(self):
        if not os.path.exists('./nrrd_file/'):
            self._image_to_nrrd()

        with open('./nrrd_file/pickle_file.txt', 'rb') as f:
            finding_list = pickle.load(f)

        df = pd.DataFrame()
        for results in finding_list:  # results是一个字典
            img = results['img']
            mask = results['mask']
            roi_value = results['roi_value']

            # 特征提取
            featureVector = _extract_features(img=img, mask=mask)
            featureVector['label'] = roi_value

            # 将提取的特征转换为DataFrame格式
            df_new = pd.DataFrame.from_dict(featureVector.values()).T
            df_new.columns = featureVector.keys()
            df = pd.concat([df, df_new])

        # 将提取的特征结果写入文件
        self.filename = f'./nrrd_file/results_{int(time.time())}.xlsx'
        with pd.ExcelWriter(self.filename) as writer:
            df.to_excel(writer, index=False)

    def data_preprocess(self, start_column_num=22):
        # 数据预处理
        self.data = pd.read_excel(self.filename).iloc[:, start_column_num:]

        # 数据集划分
        # test_size 7 3 分, random_state 每次运行都是一个稳定的结果
        # data_train, data_test = train_test_split(self.data, test_size=0.3, random_state=15)
        data_train, data_test = train_test_split(self.data, test_size=0.3)
        self.data_train_a = data_train[data_train['label'] == 0]
        self.data_train_b = data_train[data_train['label'] == 1]
        self.data_test_a = data_test[data_test['label'] == 0]
        self.data_test_b = data_test[data_test['label'] == 1]
        print(self.data_train_a.shape)
        print(self.data_train_b.shape)
        print(self.data_test_a.shape)
        print(self.data_test_b.shape)

    def t_test(self):
        # t检验特征筛选
        index = []
        for colName in self.data.columns[:]:
            if levene(self.data_train_a[colName], self.data_train_b[colName])[1] > 0.05:
                if ttest_ind(self.data_train_a[colName], self.data_train_b[colName])[1] < 0.05:
                    index.append(colName)
            else:
                if ttest_ind(self.data_train_a[colName], self.data_train_b[colName], equal_var=False)[1] < 0.05:
                    index.append(colName)
        print('t检验后, 筛选出', len(index), '个特征')
        print('这些特征分别是:', index)

        # t检验后训练集数据整理
        data_train_a = self.data_train_a[index]
        data_train_b = self.data_train_b[index]
        data_train = pd.concat([data_train_a, data_train_b])
        data_train = shuffle(data_train)
        data_train.index = range(len(data_train))  # 打乱后重新赋值索引
        self.X_train = data_train[data_train.columns[:-1]]

        # 注意下面两行在训练集与测试集上的区别; 因为是同一组数据, 标准化的方法要一致
        scaler = StandardScaler()
        scaler.fit(self.X_train)

        self.X_train = scaler.transform(self.X_train)
        self.X_train = pd.DataFrame(self.X_train)
        self.X_train.columns = index[:-1]
        self.y_train = data_train['label']

        # t检验后测试集数据整理
        data_test_a = self.data_test_a[index]
        data_test_b = self.data_test_b[index]
        data_test = pd.concat([data_test_a, data_test_b])
        data_test = shuffle(data_test)
        data_test.index = range(len(data_test))  # 打乱后重新赋值索引
        self.X_test = data_test[data_test.columns[:-1]]
        self.X_test = scaler.transform(self.X_test)  # 这里注意, 使用的是同一把尺子
        self.X_test = pd.DataFrame(self.X_test)
        self.X_test.columns = index[:-1]
        self.y_test = data_test['label']

    def lasso(self):
        # LASSO特征筛选
        self.alphas = np.logspace(-4, 1, 50)  # alpha就是损失函数的拉姆达， 从10的-4次方到10的一次方均匀地取50个值
        self.model_lassoCV = LassoCV(alphas=self.alphas, max_iter=100000).fit(self.X_train, self.y_train)
        self.coef = pd.Series(self.model_lassoCV.coef_, index=self.X_train.columns)  # coef_代表LASSO计算出来的每个特征的系数
        print('最优alpha值为:', self.model_lassoCV.alpha_)  # alpha代表选出来的最优alpha值
        print('%s %d' % ('Lasso选择出的系数不为0的特征的个数:', sum(self.coef != 0)))
        print('Lasso选择出的系数不为0的特征:\n', self.coef[self.coef != 0], sep='')
        self.index = self.coef[self.coef != 0].index
        self.X_train_raw = self.X_train
        self.X_train = self.X_train[self.index]
        self.X_test = self.X_test[self.index]

    def pca(self):
        # 主成分分析
        model_pca = PCA(n_components=0.99)  # n_components代表降维后的特征至少能解释原始特征99%的方差
        model_pca.fit(self.X_train)
        self.X_train = model_pca.fit_transform(self.X_train)
        self.X_test = model_pca.transform(self.X_test)
        print(f'训练集中PCA降维前的数据形状为:{self.X_train_raw.shape}, PCA降维后的数据形状为:{self.X_train.shape}')
        print('训练集中降维后每个特征的方差', model_pca.explained_variance_)
        print('训练集中降维后每个特征能够解释原始信息的百分比', model_pca.explained_variance_ratio_)
        print('训练集中降维后所有特征总共能解释原始信息的百分比', sum(model_pca.explained_variance_ratio_))

    def random_forest(self):
        # 随机森林分类器
        self.model = RandomForestClassifier(
            n_estimators=200,  # 设置随机森林中一共有多少颗树, 默认是100
            criterion='entropy',  # 随机森林分类器应用什么标准来判定, 主要有两种: gini指数和熵, 默认是gini
            random_state=20,  # 每次运行都是一个稳定的结果, 默认是None
            class_weight='balanced')  # 类别的权重, 目的是让0类和1类的数据平衡, 默认是None
        self.model.fit(self.X_train, self.y_train)
        print('测试集上的平均准确率为:', self.model.score(self.X_test, self.y_test))
        print('测试集的正确结果为:', list(self.y_test))
        print('对测试集进行结果预测的展示:', self.model.predict(self.X_test))
        print('测试集中每种预测结果的概率展示:', self.model.predict_proba(self.X_test))
        print('在拟合过程中用到了多少特征:', self.model.n_features_in_)
        print('用到的每个特征的权重:', self.model.feature_importances_)
        print('展示随机森林模型的各个参数:', self.model.get_params())

    def svm(self):
        # 支持向量机分类器
        self.model = SVC(kernel='rbf',  # rbf：径向基核函数，常用的核函数有：
                         # 线性核函数（linear），多项式核函数（poly），径向基核函数（rbf），sigmoid核函数
                         gamma='scale',  # gamma定义了单个样本的影响范围，gamma越大，支持向量越多，也可以写auto、浮点数
                         probability=True,  # 后面可以看分到哪一类的概率
                         class_weight='balanced',  # 类别的权重, 目的是让0类和1类的数据平衡, 默认是None
                         random_state=20,  # 每次运行都是一个稳定的结果, 默认是None
                         )
        self.model.fit(self.X_train, self.y_train)
        print('测试集上的平均准确率为:', self.model.score(self.X_test, self.y_test))
        print('测试集的正确结果为:', list(self.y_test))
        print('对测试集进行结果预测的展示:', self.model.predict(self.X_test))
        print('测试集中每种预测结果的概率展示:', self.model.predict_proba(self.X_test))
        print('在拟合过程中用到了多少特征:', self.model.n_features_in_)
        print('展示支持向量机模型的各个参数:', self.model.get_params())

    def feature_weight_figure(self):
        # 特征权重图
        plt.figure(figsize=(12, 10), dpi=80)
        x_values = np.arange(len(self.index))
        y_values = self.coef[self.coef != 0]
        plt.bar(
            x_values, y_values,  # 横向bar使用：barh
            color='lightblue',  # 设置bar的颜色
            edgecolor='black',  # 设置bar边框颜色
            alpha=0.8,  # 设置不透明度
        )
        plt.xticks(
            x_values, self.index,
            rotation='45',  # 旋转xticks
            ha='right',  # xticks的水平对齐方式
            va='top',  # xticks的垂直对齐方式
            fontsize=12
        )
        plt.xlabel("feature")  # 横轴名称
        plt.ylabel("weight")  # 纵轴名称
        plt.title('Weight of features', fontsize=22)
        plt.show()

    def feature_correlation_figure(self):
        # 特征相关性热度图
        plt.figure(figsize=(12, 10), dpi=80)
        sns.heatmap(
            self.X_train.corr(),  # 计算特征间的相关性
            xticklabels=self.X_train.corr().columns,
            yticklabels=self.X_train.corr().columns,
            cmap='RdYlGn',
            center=0.5,
            annot=True
        )
        plt.title('Correlogram of features', fontsize=22)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    def lambda_selection_figure(self):
        # LASSO模型中Lambda选值图
        MSEs = self.model_lassoCV.mse_path_
        MSEs_mean = np.apply_along_axis(np.mean, 1, MSEs)
        MSEs_std = np.apply_along_axis(np.std, 1, MSEs)

        plt.figure(dpi=80)
        plt.errorbar(
            self.model_lassoCV.alphas_, MSEs_mean,  # x, y数据，一一对应，这里用的是alphas_，而不是alpha_
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
        plt.axvline(self.model_lassoCV.alpha_, color='black', ls="--")
        plt.xlabel('Lambda')
        plt.ylabel('MSE')
        plt.title('Selection of lambda', fontsize=22)
        plt.show()

    def lambda_coefficient_figure(self):
        # 特征系数随Lambda变化曲线
        coef = self.model_lassoCV.path(self.X_train_raw, self.y_train, alphas=self.alphas, max_iter=100000)[1].T
        plt.figure(dpi=80)
        plt.semilogx(self.model_lassoCV.alphas_, coef, '-')
        plt.axvline(self.model_lassoCV.alpha_, color='black', ls="--")
        plt.xlabel('Lambda')
        plt.ylabel('Coefficients')
        plt.title('Coefficient with lambda', fontsize=22)
        plt.show()

    def roc_curve_figure(self):
        y_test_probs = self.model.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_test_probs[:, 1], pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr, marker='o')
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")

        y_test_pred = self.model.predict(self.X_test)
        auc_score = roc_auc_score(self.y_test, y_test_pred)

        # ROC曲线的AUC
        print('ROC曲线的AUC为:', auc_score)
        #  精确度（Precision），敏感度（Sensitivity），特异度（Specificity）等输出
        #  将“1”类作为“阳性”时，“1”类的recall就是Sensitivity，“0”类的recall即为Specificity
        print('分类报告如下:\n', classification_report(self.y_test, y_test_pred), sep='')

        plt.title('ROC Curve', fontsize=22)
        plt.show()


if __name__ == '__main__':
    ra = RadiomicsAnalysis()
    ra.svm_route()
