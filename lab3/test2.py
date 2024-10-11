import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold
import numpy as np

def evaluate(data:pd.DataFrame,dt:DecisionTreeClassifier):
    '''评估函数,使用F1进行评估'''
    data_length = len(data)
    data=data.values
    X = [x[0:-1] for x in data]
    #print(X)
    Y = [y[-1] for y in data]
    TP,FP,FN,TN=0,0,0,0
    for i in range(data_length):
        x=dt.predict([X[i]])
        y=Y[i]
        if (x==1 and y==1): TP+=1
        if (x==0 and y==1): FN+=1
        if (x==1 and y==0): FP+=1
        if (x==0 and y==0): TN+=1
    F1=2*TP/(2*TP+FP+FN)
    return F1
def create_data(path):
    '''初始化处理数据的函数'''
    df = pd.read_excel(path)
    # 获取数据集和每个维度的名称
    df = df.drop(['nameid'], axis=1)
    re = [0,10000,20000,30000,40000,50000]
    df['revenue']=pd.cut(df['revenue'],re,labels=False)
    datasets = df.values
    labels = df.columns.values
    return pd.DataFrame(datasets, columns=labels)
def parameter_test(i,train_data):
    '''计算一次10折交叉验证的效果的函数'''
    kf=KFold(n_splits=10,shuffle=False)
    train_data=np.array(train_data)
    e=0
    tree_clf = DecisionTreeClassifier(max_depth=i)
    for train_index,valid_index in kf.split(train_data):
        train,valid=train_data[train_index],train_data[valid_index]
        X = np.array([x[0:-1] for x in train])
        Y = np.array([y[-1] for y in train])
        tree_clf.fit(X,Y)
        valid=pd.DataFrame(valid)
        e+=evaluate(valid,tree_clf)
    return e/10
def find_parameter(train_data):
    '''寻找最佳参数的函数'''
    m=0
    i_save=0
    for i in range(1,20):
        test=parameter_test(i,train_data=train_data)
        if (test>m):
            m=test
            i_save=i
    return i_save,m
#初始化处理数据
train_data = create_data(r"银行借贷数据集train.xls")
test_data = create_data(r"银行借贷数据集test.xls")
feature =['profession','education','house_loan','car_loan',\
          'married','child','revenue']
classname =['unapprove','approve']
train_data=train_data.values
X = [x[0:-1] for x in train_data]
Y = [y[-1] for y in train_data]
#寻找最佳参数
para=find_parameter(train_data=train_data)[0]
print("计算的最佳参数为:"+str(para))
#训练最佳参数下的决策树和绘制相应的图像
tree_clf = DecisionTreeClassifier(max_depth=para)
tree_clf.fit(X, Y)
print("评估效果(F1)"+str(evaluate(test_data,tree_clf)))
dot_data=export_graphviz(
            tree_clf,
            out_file=None,
            feature_names=feature,
            class_names=classname,
            rounded=True,
            filled=True,
    special_characters=True)
dot_data=dot_data.replace('\n','')
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write_png(r"loan.png")