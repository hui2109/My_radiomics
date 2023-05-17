#导入常用R包
library(glmnet)
library(rms)
library(foreign)
library(ggplot2)
library(pROC)
#设置种子为了保证每次结果都一样
set.seed(888)

data <- read.csv("csv/TotalOMICS.csv")
nn=0.8
print(paste('总样本数:',length(data[,1])))

sub<-sample(1:nrow(data),round(nrow(data)*nn))  # 从1到177，随机取142个数
trainOmics<-data[sub,]  # 取0.8的数据做训练集
testOmics<-data[-sub,]  # 取0.2的数据做测试集
print(paste('训练集样本数:',length(trainOmics[,1])))
print(paste('测试集样本数:',length(testOmics[,1])))
write.csv(trainOmics,"csv/trainOmics.csv",row.names = FALSE )
write.csv(testOmics,"csv/testOmics.csv",row.names = FALSE )

trainClinic<-read.csv("./csv/trainClinic.csv",fileEncoding = "UTF-8-BOM")
trainClinic<- data.frame(trainClinic)
#打印临床特征名
names(trainClinic)
# 查看临床特征的基本信息
summary(trainClinic)
str(trainClinic)

# age变量转换成数值型
trainClinic$Age <- as.numeric(trainClinic$Age)

model.Clinic<-glm(Label~Age,data = trainClinic,family=binomial(link='logit'))
summary(model.Clinic)

# 查看模型在训练集的情况
probClinicTrain<-predict.glm(object=model.Clinic,newdata=trainClinic,type = "response")
predClinicTrain<-ifelse(probClinicTrain>=0.5,1,0)  # 将预测概率大于0.5的视为阳性

# 计算模型精度
error=predClinicTrain-trainClinic$Label
# accuracy:判断正确的数量占总数的比例
accuracy=(nrow(trainClinic)-sum(abs(error)))/nrow(trainClinic)

# precision:即真阳性率，真实值预测值全为1 / 预测值全为1 --- 提取出的正确信息条数/提取出的信息条数 
precision=sum(trainClinic$Label & predClinicTrain)/sum(predClinicTrain)

# recall:即敏感度，真实值预测值全为1 / 真实值全为1 --- 提取出的正确信息条数 /样本中的信息条数 
recall=sum(predClinicTrain & trainClinic$Label)/sum(trainClinic$Label)

# P和R指标有时候会出现的矛盾的情况，这样就需要综合考虑他们，最常见的方法就是F-Measure（又称为F-Score）
# F-Measure是Precision和Recall加权调和平均，是一个综合评价指标 
F_measure=2*precision*recall/(precision+recall)

# 输出以上各结果
# precision和recall和F_measure是预测Hgg的值
# 可以模型预测HGG的能力比较强，但是预测Lgg的能力比较弱
print(paste('准确率accuracy:',accuracy))
print(paste('精确率precision:',precision))
print(paste('召回率recall:',recall))
print(paste('F_measure:',F_measure))
# 绘制四格表
table(trainClinic$Label,predClinicTrain)

# 查看模型在测试集的情况
testClinic<-read.csv("./csv/testClinic.csv",fileEncoding = "UTF-8-BOM")
testClinic$Age <- as.numeric(testClinic$Age)
probClinicTest<-predict.glm(object=model.Clinic,newdata=testClinic,type = "response")

predClinicTest<-ifelse(probClinicTest>=0.5,1,0)  # 将预测概率大于0.5的视为阳性

error=predClinicTest-testClinic$Label
accuracy=(nrow(testClinic)-sum(abs(error)))/nrow(testClinic)
precision=sum(testClinic$Label & predClinicTest)/sum(predClinicTest)
recall=sum(predClinicTest & testClinic$Label)/sum(testClinic$Label)
F_measure=2*precision*recall/(precision+recall)    

print(paste('准确率accuracy:',accuracy))
print(paste('精确率precision:',precision))
print(paste('召回率recall:',recall))
print(paste('F_measure:',F_measure))
# 绘制四格表
table(testClinic$Label,predClinicTest)

# 加载通过T检验后的数据进行lasso特征筛选
tData_train <- read.csv("csv/tData_train.csv",fileEncoding = "UTF-8-BOM")
dim(tData_train)  # 输出数据维度
Y <- as.data.frame(tData_train$label)
Y <- model.matrix(~.,data=Y)[,-1]  # [,-1]是为了去掉截距
# 除去因变量，提取自变量
yavars <- names(tData_train) %in% c("label","index")  # 将列名称为["label","index"]的列名称设为True，否则设为False
X <- as.data.frame(tData_train[!yavars])
X <- model.matrix(~.,data=X)[,-1]
# Lasso回归
fit <- glmnet(X,Y, alpha=1, family = "binomial")

# 绘制相关图像
plot(fit, xvar="lambda", label=TRUE)

cv.fit <- cv.glmnet(X,Y, alpha=1,nfolds = 10,family="binomial")
plot(cv.fit)
abline(v=log(c(cv.fit$lambda.min, cv.fit$lambda.lse)), lty=2)

plot(cv.fit$glmnet.fit,xvar="lambda")
abline(v=log(cv.fit$lambda.1se), lty=2,)

#如果取1倍标准误时,获取筛选后的特征
lambda = cv.fit$lambda.1se
Coefficients <- coef(fit, s = lambda)
Active.Index <- which(Coefficients != 0)
Active.Coefficients <- Coefficients[Active.Index]
Active.Index
Active.Coefficients
row.names(Coefficients)[Active.Index]

# 建立公式
formulalse <-as.formula(label ~wavelet.LHL_glcm_Imc2+wavelet.LHL_glszm_ZoneEntropy+wavelet.HLL_glszm_ZoneEntropy+wavelet.LLL_firstorder_90Percentile+wavelet.LLL_firstorder_RootMeanSquared)
# 逻辑回归
model.Omics <- glm(formula=formulalse,data=tData_train,family=binomial(link="logit"))
#查看结果
summary(model.Omics)

# 查看模型在单纯影像组学[训练集]的情况
probOmicsTrain<-predict.glm(object =model.Omics,newdata=tData_train,type = "response")
predOmicsTrain<-ifelse(probOmicsTrain>=0.5,1,0)
error=predOmicsTrain-tData_train$label
accuracy=(nrow(tData_train)-sum(abs(error)))/nrow(tData_train)

precision=sum(tData_train$label & predOmicsTrain)/sum(predOmicsTrain)
recall=sum(predOmicsTrain & tData_train$label)/sum(tData_train$label)
F_measure=2*precision*recall/(precision+recall)    
print(paste('准确率accuracy:',accuracy))
print(paste('精确率precision:',precision))
print(paste('召回率recall:',recall))
print(paste('F_measure:',F_measure))

table(tData_train$label,predOmicsTrain)

# 查看模型在单纯影像组学[测试集]的情况
tData_test <- read.csv("csv/testOmics.csv",fileEncoding = "UTF-8-BOM")

probOmicsTest<-predict.glm(object =model.Omics,newdata=tData_test,type = "response")
predOmicsTest<-ifelse(probOmicsTest>=0.5,1,0)
error=predOmicsTest-tData_test$label
accuracy=(nrow(tData_test)-sum(abs(error)))/nrow(tData_test)

precision=sum(tData_test$label & predOmicsTest)/sum(predOmicsTest)
recall=sum(predOmicsTest & tData_test$label)/sum(tData_test$label)
F_measure=2*precision*recall/(precision+recall)    
print(paste('准确率accuracy:',accuracy))
print(paste('精确率precision:',precision))
print(paste('召回率recall:',recall))
print(paste('F_measure:',F_measure))

table(tData_test$label,predOmicsTest)

# 结合临床特征和影像组学进行建模
# 流程如下：  1.通过上面Lasso回归计算训练集的得分
#             2.把这个得分结合Age特征一起建立模型

# 读取经过T检验筛选后的影像组学训练集文件
tData_train2 <- read.csv("csv/tData_train.csv",fileEncoding = "UTF-8-BOM")
# 把临床特征的Age放到影像组学中作为新的一列
# 放的时候是根据病例名一一对应的
tData_train2$Age <- trainClinic$Age
for (i in trainClinic$index){
  if(tData_train2[tData_train2$index == i,]$index == i){
    Age <- trainClinic[trainClinic$index == i,]$Age
    tData_train2[tData_train2$index == i,]$Age <- Age
  }
}

# 对测试集也这样处理
tData_test2 <- read.csv("csv/testOmics.csv",fileEncoding = "UTF-8-BOM")
testClinic2<-read.csv("./csv/testClinic.csv",fileEncoding = "UTF-8-BOM")
testClinic2<- data.frame(testClinic2)
testClinic2$Age <- as.numeric(testClinic2$Age)
columns <- colnames(tData_train2)[1:length(tData_train2)-1]  # 经过T检验筛选后的562个特征+index+label，不要年龄特征
tData_test2 <- as.data.frame(tData_test2[,columns])
tData_test2$Age <- testClinic2$Age
for (i in testClinic2$index){
  if(tData_test2[tData_test2$index == i,]$index == i){
    Age <- testClinic2[testClinic2$index == i,]$Age
    
    tData_test2[tData_test2$index == i,]$Age <- Age
  }
}

# 分别计算影像组学得分（RS）
# 单独取出label
y_vad <-as.data.frame(tData_test2$label)
y_vad <- model.matrix(~.,data=y_vad)[,-1]
# 除去因变量，提取自变量
yavars<-names(tData_test2) %in% c("label", 'index', 'Age')  # T检验筛选后的564个特征，不要index、label、age
x_vad <- as.data.frame(tData_test2[!yavars])
x_vad <- model.matrix(~.,data=x_vad)[,-1]

x_dev <- X  # X为训练集t检验后提取的影像组学特征
y_dev <- Y  # Y为训练集标签数据

# fit是lassoCV模型
# RS:影像组学得分
tData_train2$RS <-predict(fit,type="link",newx=x_dev,newy=y_dev,s=cv.fit$lambda.1se)  # 训练集的RS
tData_test2$RS<-predict(fit,type="link",newx=x_vad,newy=y_vad,cv.fit$lambda.1se)  # 测试集的RS

# 建模
# 已经算好RS得分，临床特征Age也加到里面，现在可以建立逻辑回归
# 因为后续要绘制诺模图，所以用rms这个包建立逻辑回归
# 通过下列例子，发现两种的逻辑回归结果是一样的
tData_train2$RS <- as.numeric(tData_train2$RS)
model.and1 = glm(label ~ RS+Age,data=tData_train2,binomial(link='logit'))
print(model.and1$coef)
print("#####")
model.and2 <- lrm(label ~ RS+Age,data=tData_train2,x=TRUE,y=TRUE)
print(model.and2$coef)  # model.and1$coef 等于 model.and2$coef

# 查看模型在临床结合影像组学的[训练集]的情况
probOmicsTrainAnd<-predict.glm(object =model.and1,newdata=tData_train2,type = "response")
predOmicsTrainAnd<-ifelse(probOmicsTrainAnd>=0.5,1,0)
error=predOmicsTrainAnd-tData_train2$label
accuracy=(nrow(tData_train2)-sum(abs(error)))/nrow(tData_train2)

precision=sum(tData_train2$label & predOmicsTrainAnd)/sum(predOmicsTrainAnd)
recall=sum(predOmicsTrainAnd & tData_train2$label)/sum(tData_train2$label)
F_measure=2*precision*recall/(precision+recall)    
print(paste('准确率accuracy:',accuracy))
print(paste('精确率precision:',precision))
print(paste('召回率recall:',recall))
print(paste('F_measure:',F_measure))

table(tData_train2$label,predOmicsTrainAnd)

# 查看模型在临床结合影像组学的[测试集]的情况
tData_test2$RS <- as.numeric(tData_test2$RS)
probOmicsTestAnd<-predict.glm(object =model.and1,newdata=tData_test2,type = "response")
predOmicsTestAnd<-ifelse(probOmicsTestAnd>=0.5,1,0)
error=predOmicsTestAnd-tData_test2$label
accuracy=(nrow(tData_test2)-sum(abs(error)))/nrow(tData_test2)

precision=sum(tData_test2$label & predOmicsTestAnd)/sum(predOmicsTestAnd)
recall=sum(predOmicsTestAnd & tData_test2$label)/sum(tData_test2$label)
F_measure=2*precision*recall/(precision+recall)    
print(paste('准确率accuracy:',accuracy))
print(paste('精确率precision:',precision))
print(paste('召回率recall:',recall))
print(paste('F_measure:',F_measure))

table(tData_test2$label,predOmicsTestAnd)

# 绘制[训练集]ROC
# 绘制单纯临床、单纯影像组学、临床+影像组学的ROC曲线
rocClinic <- roc(trainClinic$Label, probClinicTrain)  # 单纯临床ROC
rocOmics <- roc(tData_train$label, probOmicsTrain)  # 单纯影像组学ROC
rocClinicAndOmics <-roc(tData_train2$label, probOmicsTrainAnd)  # 临床+影像组学ROC
rocClinic
rocOmics
rocClinicAndOmics

# 先绘制1条ROC曲线
plot(rocClinic, 
     print.auc=TRUE, # 图像上输出AUC的值
     print.auc.x=0.4, print.auc.y=0.6, # 设置AUC值坐标为（x，y）
     auc.polygon=TRUE, # 将ROC曲线下面积转化为多边形
     auc.polygon.col="#fff7f7",  # 设置ROC曲线下填充色
     grid=FALSE,
     print.thres=FALSE, # 图像上输出最佳截断值
     main=" Train ROC curves",  # 添加图形标题
     col="red",    # 设置ROC曲线颜色
     legacy.axes=TRUE,# 使x轴从0到1，表示为1-特异度
     xlim=c(1,0),
     mgp=c(1.5, 1, 0),
     lty=3)   
# 再添加1条ROC曲线
plot.roc(rocOmics,
         add=TRUE, # 增加曲线,
         col="green", # 设置ROC曲线颜色
         print.thres=FALSE, # 图像上输出最佳截断值
         print.auc=TRUE,   # 图像上输出AUC
         print.auc.x=0.4,print.auc.y=0.5,# AUC的坐标为（x，y）
         lty=2)

# 再添加1条ROC曲线
plot.roc(rocClinicAndOmics,
         add=TRUE, # 增加曲线,
         col="blue", # 设置ROC曲线颜色
         print.thres=FALSE, # 图像上输出最佳截断值
         print.auc=TRUE,   # 图像上输出AUC
         print.auc.x=0.4,print.auc.y=0.4,# AUC的坐标为（x，y）
         lty=1)


# 添加图例
legend(0.45, 0.30,  # 图例位置x，y
       bty = "n",   # 图例样式
       legend=c("Clinical model","Radiomics signature","Clinical and Radiomics"),  # 添加分组
       col=c("red","green","blue"),  # 颜色跟前面一致
       lwd=1,
       lty=c(3,2,1))  # 线条粗细

# 绘制[测试集]ROC
rocClinicTest <- roc(testClinic$Label, probClinicTest) 
rocOmicsTest <- roc(tData_test$label, probOmicsTest)
rocClinicAndOmicsTest <-roc(tData_test2$label, probOmicsTestAnd)
rocClinicTest
rocOmicsTest
rocClinicAndOmicsTest

plot(rocClinicTest, 
     print.auc=TRUE, # 图像上输出AUC的值
     print.auc.x=0.4, print.auc.y=0.6, # 设置AUC值坐标为（x，y）
     auc.polygon=TRUE, # 将ROC曲线下面积转化为多边形
     auc.polygon.col="#fff7f7",  # 设置ROC曲线下填充色
     grid=FALSE,
     print.thres=FALSE, # 图像上输出最佳截断值
     main=" Test ROC curves",  # 添加图形标题
     col="red",    # 设置ROC曲线颜色
     legacy.axes=TRUE,# 使x轴从0到1，表示为1-特异度
     xlim=c(1,0),
     mgp=c(1.5, 1, 0),
     lty=3)   
# 再添加1条ROC曲线
plot.roc(rocOmicsTest,
         add=TRUE, # 增加曲线,
         col="green", # 设置ROC曲线颜色
         print.thres=FALSE, # 图像上输出最佳截断值
         print.auc=TRUE,   # 图像上输出AUC
         print.auc.x=0.4,print.auc.y=0.5,# AUC的坐标为（x，y）
         lty=2)

# 再添加1条ROC曲线
plot.roc(rocClinicAndOmicsTest,
         add=TRUE, # 增加曲线,
         col="blue", # 设置ROC曲线颜色
         print.thres=FALSE, # 图像上输出最佳截断值
         print.auc=TRUE,   # 图像上输出AUC
         print.auc.x=0.4,print.auc.y=0.4,# AUC的坐标为（x，y）
         lty=1)


# 添加图例
legend(0.45, 0.30,  # 图例位置x，y
       bty = "n",   # 图例样式
       legend=c("Clinical model","Radiomics signature","Clinical and Radiomics"),  # 添加分组
       col=c("red","green","blue"),  # 颜色跟前面一致
       lwd=1,
       lty=c(3,2,1))  # 线条粗细

# 对临床+影像组学模型绘制诺模图、校准曲线、DCA曲线
# 绘制[诺模图]
library(rms)
formula <- as.formula(label ~ Age+RS)
# 数据打包
dd = datadist(tData_train2)
options(datadist="dd")
fitnomogram <- lrm(formula,data=tData_train2, x=TRUE, y=TRUE)

nom1 <- nomogram(fitnomogram,  # 模型对象
                 fun=function(x)1/(1+exp(-x)),  # 保持不变，logistic固定
                 lp=F,  # 是否显示线性概率
                 fun.at=c(0.1,0.2,0.5,0.7,0.9),  # 预测结果坐标轴
                 funlabel="Risk")  # 坐标轴名称
#可以使用Cairo导出pdf
#library(Cairo)
#CairoPDF("nomogram.pdf",width=6,height=6)
plot(nom1)

# 绘制[校准曲线]
cal1 <- calibrate(fitnomogram, cmethod="hare", method="boot", B=1000)
plot(cal1, xlim=c(0,1.0), ylim=c(0,1.0),
     xlab = "Predicted Probability", 
     ylab = "Observed Probability",
     legend = FALSE,subtitles = FALSE)

# abline对角线
abline(0,1,col = "black",lty = 2,lwd = 2)
# 再画一条模型预测的实际曲线
lines(cal1[,c("predy","calibrated.orig")], 
      type = "l",lwd = 2,col="red",pch =16)
# 再画一条模型Bias-corrected是校准曲线
lines(cal1[,c("predy","calibrated.corrected")], 
      type = "l",lwd = 2,col="green",pch =16)
legend(0.55,0.35,
       c("Ideal","Apparent","Bias-corrected"),
       lty = c(2,1,1),
       lwd = c(2,1,1),
       col = c("black","red","green"),
       bty = "n") # "o"为加边框

# 绘制[DCA曲线] (临床决策曲线)
# 把三个模型的临床决策曲线都绘制在一起
library(rmda)
formulClinic<-as.formula(Label~Age)

formulaOmics <- as.formula(label ~wavelet.LHL_glcm_Imc2+wavelet.LHL_glszm_ZoneEntropy+wavelet.HLL_glszm_ZoneEntropy+wavelet.LLL_firstorder_90Percentile+wavelet.LLL_firstorder_RootMeanSquared)

formulaAnd <- as.formula(label~Age+RS)

model_Clinic <- decision_curve(formulClinic, data=trainClinic, family=binomial(link='logit'), thresholds=seq(0,1,by=0.01), confidence.intervals=0.95, study.design = 'case-control', population.prevalence=0.3)
model_Omics <- decision_curve(formulaOmics, data=tData_train, family=binomial(link='logit'), thresholds=seq(0,1,by=0.01), confidence.intervals=0.95, study.design = 'case-control', population.prevalence=0.3)
model_And <- decision_curve(formulaAnd, data=tData_train2, family=binomial(link='logit'), thresholds=seq(0,1,by=0.01), confidence.intervals=0.95, study.design = 'case-control', population.prevalence=0.3)

model_all <- list(model_Clinic,model_Omics,model_And)
plot_decision_curve(model_all,curve.names=c('Clinical model','Radiomics signature','Clinical and Radiomics'),
                    xlim=c(0,0.8),
                    cost.benefit.axis=F,col=c('green','red','blue'),
                    confidence.intervals = F,
                    standardize = F,
                    legend.position="bottomleft")
# 结束