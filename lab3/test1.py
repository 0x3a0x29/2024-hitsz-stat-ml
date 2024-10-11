import numpy as np
import pandas as pd
from math import log
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

# 定义节点类 二叉树
class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label:': self.label, 'feature': self.feature, 'tree': self.tree}

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        if features[self.feature] not in self.tree:
            key_feature = list(self.tree.keys())  # 储存特征出现过的取值
            x = np.random.randint(0, len(key_feature))  # 随机生成一个数字
            random_key = key_feature[x]
            return self.tree[random_key].predict(features)
        return self.tree[features[self.feature]].predict(features)
class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}
    # 熵
    def calc_ent(self, datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p/data_length)*log(p/data_length, 2) for p in label_count.values()])
        return ent
    # 经验条件熵
    def cond_ent(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p)/data_length)*self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent
    # 信息增益
    def info_gain(self, ent, cond_ent):
        return ent - cond_ent
    # 关于特征的熵
    def calc_ent2(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = 0
            feature_sets[feature]+=1
        cond_ent = -sum([p/data_length*log(p/data_length, 2) for p in feature_sets.values()])
        return cond_ent
    #信息增益比
    def info_ratio(self,info,ent):
        if (ent==0):
            return 0
        else:
            return info/ent
    #返回信息增益最大的特征
    def info_ratio_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            c_info_ratio=self.info_ratio(c_info_gain,self.calc_ent2(datasets,axis=c))
            best_feature.append((c, c_info_ratio))
        # 比较大小
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_
    #训练决策树
    def train(self, train_data):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]

        if len(y_train.value_counts()) == 1:
            return Node(root=True,
                        label=y_train.iloc[0])

        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        max_feature, max_info_gain = self.info_ratio_train(np.array(train_data))
        max_feature_name = features[max_feature]

        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)

        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)

            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
        
        return node_tree

    
    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)
def evaluate(data:pd.DataFrame,dt:DTree):
    '''评估函数(P,R,F1)'''
    data_length = len(data)
    df_test=data.iloc[:,:-1]
    df_result=data.iloc[:,-1]
    TP,FP,FN,TN=0,0,0,0
    for i in range(data_length):
        x=dt.predict(np.array(df_test.iloc[i]))
        y=df_result.iloc[i]
        if (x==1 and y==1): TP+=1
        if (x==0 and y==1): FN+=1
        if (x==1 and y==0): FP+=1
        if (x==0 and y==0): TN+=1
    P=TP/(TP+FP)
    R=TP/(TP+FN)
    F1=2*TP/(2*TP+FP+FN)
    return P,R,F1

train_data = create_data(r"银行借贷数据集train.xls")
test_data = create_data(r"银行借贷数据集test.xls")
dt = DTree(epsilon=0.011)#初始化决策树
tree = dt.fit(train_data)#训练决策树
P,R,F1=evaluate(test_data,dt)#评估决策树
print("准确率为:"+str(P)+"召回率为:"+str(R)+"F1为:"+str(F1))