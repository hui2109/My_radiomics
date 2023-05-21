# 将所有R代码封装为R函数
# 每个R函数独立放在一个.R文件中
# 通过loadfn方法载入R函数

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, globalenv
from rpy2.robjects import r
from sklearn.model_selection import train_test_split

from loadfn import loadfn

# 1、导入R包
import_packages = loadfn('./modified_R/import_packages.R')
import_packages()

# 2、设置种子，为了保证每次R的结果都一样
r['set.seed'](888)

# 3、提取影像组学特征
# 略

# 4、读取提取的影像组学特征，按一定比例分成训练集和测试集，并保存为csv文件
# 读取提取的影像组学特征
data = pd.read_csv('./csv/TotalOMICS.csv', encoding='utf-8')
# 按一定比例分成训练集和测试集
data_train, data_test = train_test_split(data, test_size=0.2, random_state=15)
# 打印数据形状
data_train_a = data_train[data_train['label'] == 0]
data_train_b = data_train[data_train['label'] == 1]
data_test_a = data_test[data_test['label'] == 0]
data_test_b = data_test[data_test['label'] == 1]
print('训练集，label为0的形状', data_train_a.shape, '\n',
      '训练集，label为1的形状', data_train_b.shape, '\n',
      '测试集，label为0的形状', data_test_a.shape, '\n',
      '测试集，label为1的形状', data_test_b.shape)
# 保存为csv文件
data_train.to_csv('./mcsv/trainOmics.csv', index=False, encoding='utf-8')
data_test.to_csv('./mcsv/testOmics.csv', index=False, encoding='utf-8')


# 5、读取临床特征，按照患者ID，生成对应第4步的训练集和测试集的csv文件，并保存
def compare(data1, data2, filename):
    # 读取两个表
    dt1 = pd.read_csv(data1, encoding='utf-8')
    dt2 = pd.read_csv(data2, encoding='utf-8')
    df = pd.DataFrame()
    dt1_name = dt1['index'].values.tolist()
    dt2_name = dt2['index'].values.tolist()

    for i in dt1_name:
        if i in dt2_name:
            dt2_row = dt2.loc[dt2['index'] == i]
            df = df.append(dt2_row)
    df.to_csv('./mcsv/' + filename + '.csv', header=True, index=False, encoding="utf-8")


# 保存文件
data_train = "./mcsv/trainOmics.csv"
data_test = "./mcsv/testOmics.csv"
data_clinic = "./csv/TotalClinic.csv"
compare(data_train, data_clinic, "trainClinic")
compare(data_test, data_clinic, "testClinic")

# 6、读取训练集的临床特征，打印临床特征名、查看临床特征的基本信息
# 读取训练集的临床特征
trainClinic = pd.read_csv('./mcsv/trainClinic.csv', encoding='utf-8', dtype={
    'Age': 'Int64'
})
# 先转换成R对象，再打印临床特征名、查看临床特征的基本信息
with (ro.default_converter + pandas2ri.converter).context():
    trainClinic = ro.conversion.get_conversion().py2rpy(trainClinic)
# 将trainClinic声明为全局变量
globalenv['trainClinic'] = trainClinic
df_names = r['names'](trainClinic)
df_summary = r['summary'](trainClinic)
print(df_names)
print(df_summary)

# 7、将训练集的临床特征中的age变量转换成数值型，并建立[临床特征]的线性回归模型
# 第6步已经转换，直接开始建立[临床特征]的线性回归模型
model_Clinic = r("glm(Label ~ Age, data = trainClinic, family = binomial(link = 'logit'))")
# 将model_Clinic声明为全局变量
globalenv['model_Clinic'] = model_Clinic

# 8、查看模型的基本信息、在训练集的情况
model_Clinic_summary = r['summary'](model_Clinic)
print(model_Clinic_summary)
# 在训练集的情况
# probClinicTrain：预测概率
probClinicTrain = r("predict.glm(object = model_Clinic, newdata = trainClinic, type = 'response')")
print(probClinicTrain)
# 将probClinicTrain声明为全局变量
globalenv['probClinicTrain'] = probClinicTrain
# predClinicTrain：预测值
predClinicTrain = r('ifelse(probClinicTrain >= 0.5, 1, 0)')
print(predClinicTrain)
