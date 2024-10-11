import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
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
tree_clf = DecisionTreeClassifier()
param_grid={'max_depth':np.arange(1,8),'max_leaf_nodes':np.arange(2,10)}
algo=GridSearchCV(estimator=tree_clf,param_grid=param_grid,cv=10)
algo.fit(X,Y)
print(algo.score(X,Y))
print(algo.best_params_)
[x_test, y_test] = loadDataSetTest()
predict=[]
numTestSamples = len(y_test)
dt = DecisionTreeClassifier()
print("未调参:")
TestModel(dt,X,Y,x_test,y_test)
dt = DecisionTreeClassifier(max_depth=algo.best_params_['max_depth'],max_leaf_nodes=algo.best_params_['max_leaf_nodes'])
print("调参:")
TestModel(dt,X,Y,x_test,y_test)