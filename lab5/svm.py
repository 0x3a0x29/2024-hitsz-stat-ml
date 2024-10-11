import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action = 'ignore')
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,validation_curve
# 标准化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
def loadDataSet(norm:bool):
    #读取train数据
    data=pd.read_csv('train-after.csv',encoding='gbk')
    #将data中4个有意义的特征值读取作为X
    X=data.loc[:,['frequence','amount','average','recently']]
    if (norm):
        X=X.apply(standardization)
    #将data中实际的分类标准读取作为Y
    Y=data.loc[:,'type']
    #返回处理的结果,完成对数据的预处理
    return [np.array(X),np.array(Y)]
def loadDataSetTest(norm:bool):
    #读取test数据
    data=pd.read_csv('test-after.csv',encoding='gbk')
    #将data中4个有意义的特征值读取作为X
    X=data.loc[:,['frequence','amount','average','recently']]
    if (norm):
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
    
linear = SVC(kernel='linear',decision_function_shape='ovo')
poly = SVC(kernel='poly',decision_function_shape='ovo')
rbf = SVC(kernel='rbf',decision_function_shape='ovo')
sigmoid = SVC(kernel='sigmoid',decision_function_shape='ovo')
#4(1)
[X, Y] = loadDataSet(True)
[x_test, y_test] = loadDataSetTest(True)
print("linear kernel:")
TestModel(linear,X,Y,x_test, y_test)
print("poly kernel:")
TestModel(poly,X,Y,x_test, y_test)
print("rbf kernel:")
TestModel(rbf,X,Y,x_test, y_test)
print("sigmoid kernel:")
TestModel(sigmoid,X,Y,x_test, y_test)
#4(2)
[X, Y] = loadDataSet(False)
[x_test, y_test] = loadDataSetTest(False)
print("linear kernel:")
TestModel(linear,X,Y,x_test, y_test)
print("poly kernel:")
TestModel(poly,X,Y,x_test, y_test)
print("rbf kernel:")
TestModel(rbf,X,Y,x_test, y_test)
print("sigmoid kernel:")
TestModel(sigmoid,X,Y,x_test, y_test)

#（3.1）高斯核函数的参数调节
[X, Y] = loadDataSet(True)
[x_test, y_test] = loadDataSetTest(True)
param_grid={'gamma':np.arange(10,30)/100}
algo=GridSearchCV(estimator=rbf,param_grid=param_grid,cv=10)
algo.fit(X,Y)
print(algo.score(X,Y))
print(algo.best_params_)
rbf1=SVC(kernel='rbf',decision_function_shape='ovo',gamma=algo.best_params_['gamma'])
TestModel(rbf1,X,Y,x_test, y_test)

#（3.2）多项式核函数参数的调节
[X, Y] = loadDataSet(True)
[x_test, y_test] = loadDataSetTest(True)
param_grid={'gamma':np.arange(1,3)/10,'degree':np.arange(1,5),'coef0':np.arange(1,5)}
algo=GridSearchCV(estimator=poly,param_grid=param_grid,cv=10)
algo.fit(X,Y)
print(algo.score(X,Y))
print(algo.best_params_)
poly1=SVC(kernel='poly',decision_function_shape='ovo',gamma=algo.best_params_['gamma'],degree=algo.best_params_['degree'],\
    coef0=algo.best_params_['coef0'])
TestModel(poly1,X,Y,x_test, y_test)

#（4） 松弛系数惩罚项C的调整
[X, Y] = loadDataSet(True)
range_C=np.logspace(-2,2,20)
range_G=np.logspace(-3,2,30)
train_C,test_C=validation_curve(
    SVC(kernel='rbf'),
    X,Y,
    param_name='C',
    param_range=range_C,
    cv=10,
    scoring='accuracy'
)
train_C,test_C=np.mean(train_C,axis=1),np.mean(test_C,axis=1)
plt.plot(range_C,train_C,color='r')
plt.plot(range_C,test_C,color='b')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
train_G,test_G=validation_curve(
    SVC(kernel='rbf',gamma=0.1),
    X,Y,
    param_name='gamma',
    param_range=range_G,
    cv=10,
    scoring='accuracy'
)
train_G,test_G=np.mean(train_G,axis=1),np.mean(test_G,axis=1)
plt.plot(range_G,train_G,color='r')
plt.plot(range_G,test_G,color='b')
plt.xscale('log')
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
[X, Y] = loadDataSet(True)
[x_test, y_test] = loadDataSetTest(True)
rbf1 = SVC(kernel='rbf',decision_function_shape='ovo',C=10.5,gamma=0.09)
print("rbf kernel:")
TestModel(rbf1,X,Y,x_test, y_test)