'''实验第一个内容'''
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
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
# 感知机模型核心算法（随机梯度下降法）
def fit(data, X_train, y_train):
    w = np.ones(len(data[0]) - 1, dtype=np.float32)
    b = 0
    l_rate = 0.1
    flag=True
    while(flag):
        for i in range(data.shape[0]):
            x_i= data[i,:-1]
            y_i= data[i,-1]
            if y_i*(np.dot(w,x_i)+b)<=0:
                w=w+l_rate*y_i*x_i
                b=b+l_rate*y_i
                flag=True
                break
            else:
                flag=False
    return w,b
#调用感知机模型做，得到w和b
[w,b]=fit(data, X, y)
print(w)
print(b)
#绘制图像
x_points = np.linspace(4, 7, 10)
y_ = -(w[0] * x_points + b) / w[1]
plt.plot(x_points, y_)
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='setosa')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()