import pandas as pd
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import globalenv, pandas2ri, r
from rpy2.robjects.packages import importr

# 检查R版本
print(rpy2.__version__)

# 导入R包
base = importr('base')
# datasets = importr('datasets')
# foreign = importr('foreign')
# ggplot2 = importr('ggplot2')
# glmnet = importr('glmnet')
# grDevices = importr('grDevices')
# Hmisc = importr('Hmisc')
# Matrix = importr('Matrix')
# methods = importr('methods')
# pROC = importr('pROC')
# rmda = importr('rmda')
# rms = importr('rms')
# stats = importr('stats')
utils = importr('utils')

# R实例
pi = r['pi']
print(pi[0])

# 运行R代码
pi = r('pi')
print(pi[0])

# 一个更长的代码
r('''
        # create a function `f`
        f <- function(r, verbose=FALSE) {
            if (verbose) {
                cat("I am calling f().\n")
            }
            return(2 * pi * r)
        }
        # call the function `f` with argument value 3
        f(3)
        ''')

r_f = globalenv['f']
print(r_f.r_repr())
print(r('.libPaths()'))

# 也可以这样获取f函数
r_f = r['f']
print(r_f.r_repr())

# 调用R函数
result = r_f(3, True)
print(result[0])

print('--------------------')

# 创建一个Vector
v = base.c(1, 2, 3)
print(v)
print(type(v))
print(list(v))

print('--------------------')

# 将pandas转换为R对象
pd_df = pd.DataFrame({'int_values': [1, 2, 3],
                      'str_values': ['abc', 'def', 'ghi']})
with (ro.default_converter + pandas2ri.converter).context():
    r_from_pd_df = ro.conversion.get_conversion().py2rpy(pd_df)
print(r_from_pd_df, type(r_from_pd_df))

print('--------------------')

# 将pandas转换为R对象（调用R函数时自动转换）
pd_df = pd.DataFrame({'int_values': [1, 2, 3],
                      'str_values': ['abc', 'def', 'ghi']})
with (ro.default_converter + pandas2ri.converter).context():
    df_summary = r['summary'](pd_df)
print(df_summary)

print('--------------------')

# 将R对象转换为pandas结构
r_df = ro.DataFrame({'int_values': ro.IntVector([1, 2, 3]),
                     'str_values': ro.StrVector(['abc', 'def', 'ghi'])})
with (ro.default_converter + pandas2ri.converter).context():
    pd_from_r_df = ro.conversion.get_conversion().rpy2py(r_df)
print(pd_from_r_df)
