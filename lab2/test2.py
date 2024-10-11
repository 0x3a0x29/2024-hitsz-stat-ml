'''实验第二个内容'''
#导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['label']=iris.target
df.columns=['sepal length', 'sepal width', 'petal length',
            'petal width','label']
#数据准备(选取前100行 setosa和versicolor两类鸢尾花数据)
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
for i in range(len(data)):
    if data[i,-1]==0:
        data[i,-1]=-1
#数据分割
# X是除最后一列外的所有列，y是最后一列
X, y = data[:, :-1], data[:, -1]
# 调用sklearn的train_test_split方法，将数据随机分为训练集和测试集
#数据分割
# X是除最后一列外的所有列，y是最后一列
X, y = data[:, :-1], data[:, -1]
# 调用sklearn的train_test_split方法，将数据随机分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,  #被划分的样本特征集
                                                    y,  #被划分的样本目标集
                                                    test_size=0.3, #测试样本占比
                                                    random_state=1) #随机数种子
# 定义感知机模型
clf = Perceptron(fit_intercept=True, max_iter=1000, shuffle=True,eta0=0.04,n_iter_no_change=400,
                     penalty='elasticnet',l1_ratio=0.01)
# 使用训练数据进行训练
clf.fit(X_train, y_train)
#计算模型的权重、截距、迭代次数
print("特征权重：", clf.coef_)  # 特征权重 w
print("截距（偏置）:", clf.intercept_)  # 截距 b
print("迭代次数:", clf.n_iter_)
#评价模型
print(clf.score(X_test, y_test))
#绘制图形，观察分类结果
x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0] * x_points + clf.intercept_) / clf.coef_[0][1]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(x_points, y_, 'r', label='sklearn Perceptron分类线')
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='setosa')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='versicolor')
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.title('Iris Perceptron classifier', fontsize=15)
plt.legend()
plt.show()