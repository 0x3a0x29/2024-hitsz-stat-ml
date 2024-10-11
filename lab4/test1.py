# 请补充完整代码
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action = 'ignore')
# 请补充完整代码
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import KFold
plt.rcParams['font.sans-serif']=['SimHei']  #解决中文显示乱码问题
plt.rcParams['axes.unicode_minus']=False
# 请给代码做注释
def loadDataSet():
    #读取train.xlsx数据
    data=pd.read_excel('北京市空气质量数据train.xlsx')
    #将数据data中的0替换成np.NaNs
    data=data.replace(0,np.NaN)
    #将data中含有np.NaN的数据删除
    data=data.dropna()
    #将data中6个有意义的特征值读取作为X
    X=data.loc[:,['PM2.5','PM10','SO2','CO','NO2','O3']]
    #将data中实际的空气质量等级读取作为Y
    Y=data.loc[:,'质量等级']
    #返回处理的结果,完成对数据的预处理
    return [np.array(X),np.array(Y)]
def loadDataSetTest():
    #读取train.xlsx数据
    data=pd.read_excel('北京市空气质量数据test.xlsx')
    #将数据data中的0替换成np.NaNs
    data=data.replace(0,np.NaN)
    #将data中含有np.NaN的数据删除
    data=data.dropna()
    #将data中6个有意义的特征值读取作为X
    X=data.loc[:,['PM2.5','PM10','SO2','CO','NO2','O3']]
    #将data中实际的空气质量等级读取作为Y
    Y=data.loc[:,'质量等级']
    #返回处理的结果,完成对数据的预处理
    return [np.array(X),np.array(Y)]
# 请补充完整残缺代码
def myKNN(testDigit, trainX, trainY, k):
    """
    参数testDigit:输入的新实例（也即测试集中每一行数据特征）
    参数trainX：训练集中所有的数据特征
    参数trainY：训练集中对应的类别
    参数k：提前设置好的值，表示k个分类实例
    返回：输入的新实例的判定类别：maxIndex
    """
    ####自行编程，完成k-近邻算法的实现#####
    numSamples = trainX.shape[0] 
    # 1.计算欧式距离
    distance = [np.sum(np.square(trainX[i]-testDigit)) for i in range(numSamples)]
    # 2.按距离进行排序
    sortedDistIndices = np.argsort(distance)
    classCount = {}  # 存放各类别的个体数量
    for i in range(k):
        # 3.按顺序读取标签
        voteLabel = trainY[sortedDistIndices[i]]
        # 4.计算该标签次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 5.查找出现次数最多的类别，作为分类结果
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex
def show(testPre,K):
    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle='-.')
    plt.xticks(K)
    plt.plot(K, testPre, marker='.')
    plt.xlabel("K")
    plt.ylabel("测试精度")
    bestK = K[testPre.index(np.max(testPre))]
    plt.title("K-近邻的加权F1值的变化折线图\n(最优参数K=%d)" % bestK)
    plt.show()
    
# 请补充完整残缺代码
[X, Y] = loadDataSet()
#X=
#Y=

###使用验证集寻找最合适的k值###
testPre=[]
K=np.arange(1,30,5)
split=10
kf = KFold(n_splits=split, shuffle=False)
for k in K:
    score=0
    avg_score=0
    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X[train_index],X[valid_index]
        Y_train, Y_valid = Y[train_index],Y[valid_index]
        numTestSamples = len(X_valid)
        predict=[]
        for i in range(numTestSamples):
            re = myKNN(X_valid[i],X_train,Y_train,k)
            predict.append(re)
        score = score + metrics.f1_score(Y_valid, predict, average='weighted')
        avg_score = score / split
    testPre.append(avg_score)
show(testPre, K)
###使用测试集评估模型###
# 请补充完整残缺代码
[x_test, y_test] = loadDataSetTest()
numTestSamples = len(y_test)
#matchCount = 
predict=[]
bestK = K[testPre.index(np.max(testPre))]
for i in range(numTestSamples):
    re = myKNN(x_test[i],X,Y,bestK)
    predict.append(re)

F1=metrics.f1_score(y_test, predict, average='weighted')
print('加权F值：', F1)
print('评价模型结果：\n', classification_report(y_test, predict))