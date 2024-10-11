
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action = 'ignore')
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# 标准化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
def loadDataSet():
    #读取train数据
    data=pd.read_csv('train-after.csv',encoding='gbk')
    #将data中4个有意义的特征值读取作为X
    X=data.loc[:,['frequence','amount','average','recently']]
    X=X.apply(standardization)
    #将data中实际的分类标准读取作为Y
    Y=data.loc[:,'type']
    #返回处理的结果,完成对数据的预处理
    return [np.array(X),np.array(Y)]
def loadDataSetTest():
    #读取test数据
    data=pd.read_csv('test-after.csv',encoding='gbk')
    #将data中4个有意义的特征值读取作为X
    X=data.loc[:,['frequence','amount','average','recently']]
    X=X.apply(standardization)
    #将data中实际的分类标准读取作为Y
    Y=data.loc[:,'type']
    #返回处理的结果,完成对数据的预处理
    return [np.array(X),np.array(Y)]
def TestModel(model,X,Y,x_test,y_test):
    predict=[]
    numTestSamples = len(y_test)
    model.fit(X,Y)
    for i in range(numTestSamples):
        re = model.predict([x_test[i]])[0]
        predict.append(re)

    AC=metrics.accuracy_score(y_test, predict)
    print('准确率：', AC)
    print('评价模型结果：\n', classification_report(y_test,predict))
[X, Y] = loadDataSet()
neigh = KNeighborsClassifier()
param_grid={'n_neighbors':np.arange(1,30)}
algo=GridSearchCV(estimator=neigh,param_grid=param_grid,cv=10)
algo.fit(X,Y)
print(algo.score(X,Y))
print(algo.best_params_)
[x_test, y_test] = loadDataSetTest()
predict=[]
numTestSamples = len(y_test)
print("未调参:")
neigh = KNeighborsClassifier()
TestModel(neigh,X,Y,x_test,y_test)
print("调参:")
neigh = KNeighborsClassifier(n_neighbors=algo.best_params_['n_neighbors'])
TestModel(neigh,X,Y,x_test,y_test)