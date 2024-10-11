# 请在数据处理部分 完成横向处的注释
import pandas as pd

info_user = pd.read_csv('test.csv', encoding='gbk')

# 提取info表的用户名和用餐时间，并按人名对用餐人数和金额进行分组求和
info_user1 = info_user['USER_ID'].value_counts()  # 获取用户所有出现的ID并获取频数，同时排序
info_user1 = info_user1.reset_index() # 重置标签
info_user1.columns = ['USER_ID', 'frequence']  # 修改标签名字

# 求出每个人的消费总金额
info_user2 = info_user[['number_consumers', "expenditure"]].groupby(info_user['USER_ID']).sum()  # 计算得到总金额和消费人数
info_user2 = info_user2.reset_index()
info_user2.columns = ['USER_ID', 'numbers', 'amount']  # 修改标签名字

# 将处理后得到的两个Pandas对象合成新的Pandas对象，合并数据
info_user_new = pd.merge(info_user1, info_user2, left_on='USER_ID', right_on='USER_ID', how='left') 

# 对合并后的数据进行处理
info_user = info_user.iloc[:, :4]  # 取infor_user的一部分
info_user = info_user.groupby(['USER_ID']).last()  # 按照ID升序排列
info_user = info_user.reset_index()  # 修改标签

# 将处理后得到的两个Pandas对象合成新的Pandas对象，合并数据
info_user_new = pd.merge(info_user_new, info_user, left_on='USER_ID', right_on='USER_ID', how='left')

print('合并后表中的空值数目：', info_user_new.isnull().sum().sum())
info_user_new = info_user_new.dropna(axis=0)  # 去除空表
info_user_new = info_user_new[info_user_new['numbers'] != 0] # 去除空表
print(info_user_new.head())

# 计算用户的人均消费额
info_user_new['average'] = info_user_new['amount']/info_user_new['numbers']
info_user_new['average'] = info_user_new['average'].apply(lambda x: '%.2f' % x)

# 计算每个客户最近一次点餐的时间距离观测窗口结束的天数
# 修改时间列，改为日期
info_user_new['LAST_VISITS'] = pd.to_datetime(info_user_new['LAST_VISITS'])
datefinally = pd.to_datetime('2016-7-31')  # 观测窗口结束时间
time = datefinally - info_user_new['LAST_VISITS']
info_user_new['recently'] = time.apply(lambda x: x.days)   # 计算时间差

# 将处理完的数据保存
info_user_new = info_user_new.loc[:, ['USER_ID', 'ACCOUNT', 'frequence', 'amount', 'average', 'recently', 'type']]
info_user_new.to_csv('test-after.csv', index=False, encoding='gbk')
print(info_user_new.head())



info_user = pd.read_csv('train.csv', encoding='gbk')

# 提取info表的用户名和用餐时间，并按人名对用餐人数和金额进行分组求和
info_user1 = info_user['USER_ID'].value_counts()  # 获取用户所有出现的ID并获取频数，同时排序
info_user1 = info_user1.reset_index() # 重置标签
info_user1.columns = ['USER_ID', 'frequence']  # 修改标签名字

# 求出每个人的消费总金额
info_user2 = info_user[['number_consumers', "expenditure"]].groupby(info_user['USER_ID']).sum()  # 计算得到总金额和消费人数
info_user2 = info_user2.reset_index()
info_user2.columns = ['USER_ID', 'numbers', 'amount']  # 修改标签名字

# 将处理后得到的两个Pandas对象合成新的Pandas对象，合并数据
info_user_new = pd.merge(info_user1, info_user2, left_on='USER_ID', right_on='USER_ID', how='left') 

# 对合并后的数据进行处理
info_user = info_user.iloc[:, :4]  # 取infor_user的一部分
info_user = info_user.groupby(['USER_ID']).last()  # 按照ID升序排列
info_user = info_user.reset_index()  # 修改标签

# 将处理后得到的两个Pandas对象合成新的Pandas对象，合并数据
info_user_new = pd.merge(info_user_new, info_user, left_on='USER_ID', right_on='USER_ID', how='left')

print('合并后表中的空值数目：', info_user_new.isnull().sum().sum())
info_user_new = info_user_new.dropna(axis=0)  # 去除空表
info_user_new = info_user_new[info_user_new['numbers'] != 0] # 去除空表
print(info_user_new.head())

# 计算用户的人均消费额
info_user_new['average'] = info_user_new['amount']/info_user_new['numbers']
info_user_new['average'] = info_user_new['average'].apply(lambda x: '%.2f' % x)

# 计算每个客户最近一次点餐的时间距离观测窗口结束的天数
# 修改时间列，改为日期
info_user_new['LAST_VISITS'] = pd.to_datetime(info_user_new['LAST_VISITS'])
datefinally = pd.to_datetime('2016-7-31')  # 观测窗口结束时间
time = datefinally - info_user_new['LAST_VISITS']
info_user_new['recently'] = time.apply(lambda x: x.days)   # 计算时间差

# 将处理完的数据保存
info_user_new = info_user_new.loc[:, ['USER_ID', 'ACCOUNT', 'frequence', 'amount', 'average', 'recently', 'type']]
info_user_new.to_csv('train-after.csv', index=False, encoding='gbk')
