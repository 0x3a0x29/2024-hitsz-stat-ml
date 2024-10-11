'''实验一'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #导入必要的库
df1 = pd.read_table("ReportCard1.txt",sep='\t',header=0) #读取文件形成Pandas对象
df2 = pd.read_table("ReportCard2.txt",sep='\t',header=0) #读取文件形成Pandas对象
df = pd.merge(df1,df2,how='inner',on='id') #将两个文件的Pandas对象合并
df=df.fillna(0) #将Pandas对象中的NaN数据替换成0
length,amount=df.shape #得到行数和列数
df.loc[:,'sum']=df.apply(lambda x:x[2:amount].sum(),axis=1) #计算学生总分
df.loc[:,'mean']=df.loc[0:length+1,'politics':'history'].mean(axis=1) #计算学生平均分
df=df.sort_values(by=['sum','id'],ascending=[False,True]) #按照学生总分排序
df.loc[:,'group']='错误数据' #创建新的列
df.loc[(df['mean']>=90)&(df['mean']<=100),'group']='优'
df.loc[(df['mean']>=80)&(df['mean']<90),'group']='良'
df.loc[(df['mean']>=70)&(df['mean']<80),'group']='中'
df.loc[(df['mean']>=60)&(df['mean']<70),'group']='及格'
df.loc[(df['mean']<60)&(df['mean']>=0),'group']='不及格' #得到对学生平均分的分组
df_boy=df[df['sex']==1] #得到男生数据对应的Pandas对象
df_girl=df[df['sex']==2] #得到女生数据对应的Pandas对象
df.to_csv("ReportCard.csv")
df_boy.to_csv("ReportCard_boys.csv")
df_girl.to_csv("ReportCard_girls.csv")  #保存文件
boy_mean = df_boy.loc[:,'politics':'history'].mean(axis=0) #计算男生各科的平均分
girl_mean = df_girl.loc[:,'politics':'history'].mean(axis=0) #计算女生各科的平均分
print('男生各科平均分:') #打印男生平均分
print(boy_mean)
print('女生各科平均分:') #打印女生平均分
print(girl_mean)
boy_group=np.array((df_boy.loc[df_boy['group']=='优'].shape[0],df_boy.loc[df_boy['group']=='良'].shape[0],\
    df_boy.loc[df_boy['group']=='中'].shape[0],df_boy.loc[df_boy['group']=='及格'].shape[0],\
        df_boy.loc[df_boy['group']=='不及格'].shape[0])) #得到对男生平均分的分组
girl_group=np.array((df_girl.loc[df_girl['group']=='优'].shape[0],df_girl.loc[df_girl['group']=='良'].shape[0],\
    df_girl.loc[df_girl['group']=='中'].shape[0],df_girl.loc[df_girl['group']=='及格'].shape[0],\
        df_girl.loc[df_girl['group']=='不及格'].shape[0])) #得到对女生平均分的分组
group=boy_group+girl_group
print('男生平均分分组如下,优:{0:d},良:{1:d},中:{2:d},及格:{3:d},不及格:{4:d}'.format(
    boy_group[0],boy_group[1],boy_group[2],boy_group[3],boy_group[4]
))
print('女生平均分分组如下,优:{0:d},良:{1:d},中:{2:d},及格:{3:d},不及格:{4:d}'.format(
    girl_group[0],girl_group[1],girl_group[2],girl_group[3],girl_group[4]                                                                    
))
print('全体学生平均分分组如下,优:{0:d},良:{1:d},中:{2:d},及格:{3:d},不及格:{4:d}'.format(
    group[0],group[1],group[2],group[3],group[4]  
)) #打印平均分分组
group=group/sum(group)*100
boy_group=boy_group/sum(boy_group)*100
girl_group=group/sum(girl_group)*100 #转换为百分比形式以便绘制饼状图
sum_score,math_score=np.array(df.loc[:,'sum']),np.array(df.loc[:,'math']) #计算得到总分和数学分数
plt.hist(sum_score,bins=10)
plt.title('hist of score')
plt.xlabel('score')
plt.ylabel('sum')
plt.savefig('hit.png')
plt.show() #直方图部分
plt.pie(group,labels=('excellent','good','medium','qualified','unqualified'),\
    colors=['purple','green','yellow','blue','red'])
plt.title('the rank of all')
plt.axis('equal')
plt.savefig('pie_sum.png')
plt.show() #全体学生饼状图部分
plt.pie(boy_group,labels=('excellent','good','medium','qualified','unqualified'),\
    colors=['purple','green','yellow','blue','red'])
plt.title('the rank of boys')
plt.axis('equal')
plt.savefig('pie_boys.png')
plt.show() #男学生饼状图部分
plt.pie(girl_group,labels=('excellent','good','medium','qualified','unqualified'),\
    colors=['purple','green','yellow','blue','red'])
plt.title('the rank of girls')
plt.savefig('pie_girls.png')
plt.axis('equal')
plt.show() #女学生饼状图部分
plt.scatter(math_score,sum_score,s=5)
plt.title('math-sum picture')
plt.xlabel('math')
plt.ylabel('sum')
plt.savefig('math_sum.png')
plt.show() #数学成绩-总成绩散点图部分