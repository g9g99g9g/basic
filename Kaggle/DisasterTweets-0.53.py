#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import time

pwd = 'D:\\Python\\20200204-Kaggle-Disaster Tweets\\04-Data-Preprocess\\'
trainData = pd.read_excel(pwd+'train.xlsx')
testData = pd.read_excel(pwd+'test.xlsx')  # 加载测试数据

full_data = [trainData, testData]
trainData.info()  # 训练数据的信息
testData.info()
trainData.sample(n=10)  # 随机查看训练数据
testData.sample(n=10)

print('\n'+'-'*10+'统计整体数据'+'-'*10)
print(trainData["target"].value_counts(normalize=True))

print('\n'+'-'*10+'各个离散变量和target的关系'+'-'*10)
f, axes = plt.subplots(2, 4, sharey=True)
sns.factorplot(x='commaloc', y='target', data=trainData, kind='bar', ax=axes[0, 0])
sns.barplot(x='langloc', y='target', data=trainData, ax=axes[0, 1])
sns.barplot(x='zeroloc', y='target', data=trainData, ax=axes[0, 2])
sns.barplot(x='langtext', y='target', data=trainData, ax=axes[0, 3])
sns.barplot(x='hashtag', y='target', data=trainData, ax=axes[1, 0])
sns.barplot(x='url', y='target', data=trainData, ax=axes[1, 1])
sns.barplot(x='atsb', y='target', data=trainData, ax=axes[1, 2])
sns.barplot(x='fullkw', y='target', data=trainData, ax=axes[1, 3])
#plt.show()

print('\n'+'-'*10+'各个连续变量和target的关系'+'-'*10)
f, axes = plt.subplots(3, 1, sharey=True)
sns.kdeplot(trainData["hashcount"][(trainData["target"] == 0) & (trainData["hashcount"].notnull())], \
            color="Red", shade=True, ax=axes[0], label='Not Disaster')
sns.kdeplot(trainData["hashcount"][(trainData["target"] == 1) & (trainData["hashcount"].notnull())], \
            color="Blue", shade=True, ax=axes[0], label='Disaster')
sns.kdeplot(trainData["urlcount"][(trainData["target"] == 0) & (trainData["urlcount"].notnull())], \
            color="Red", shade=True, ax=axes[1], label='Not Disaster')
sns.kdeplot(trainData["urlcount"][(trainData["target"] == 1) & (trainData["urlcount"].notnull())], \
            color="Blue", shade=True, ax=axes[1], label='Disaster')
sns.kdeplot(trainData["length"][(trainData["target"] == 0) & (trainData["length"].notnull())], \
            color="Red", shade=True, ax=axes[2], label='Not Disaster')
sns.kdeplot(trainData["length"][(trainData["target"] == 1) & (trainData["length"].notnull())], \
            color="Blue", shade=True, ax=axes[2], label='Disaster')
#plt.show()

print('\n'+'-'*10+'#hashtag、URL和target的关系'+'-'*10)
f, axes = plt.subplots(1, 1, sharey=True)
sns.pointplot(x='hashtag', y='target', hue='url', data=trainData, ax=axes)
#plt.show()

print('\n'+'-'*10+'用箱型图来检测异常值或离群点'+'-'*10)
f, axes = plt.subplots(1, 2, sharey=True)
sns.boxplot(y='length', data=trainData, ax=axes[0])
sns.boxplot(y='hashcount', data=trainData, ax=axes[1])
#plt.show()

print('\n'+'-'*10+'查看训练集和测试集中的缺失值情况'+'-'*10)
print(trainData.isnull().sum())
print(testData.isnull().sum())

print('\n'+'-'*10+'特征编码'+'-'*10)
# 将keyword分类变量转换为数值型
from sklearn.preprocessing import LabelEncoder
keyword_label_train = LabelEncoder()
trainData['fullkw'] = keyword_label_train.fit_transform(trainData['fullkw'].values)
keyword_label_test = LabelEncoder()
testData['fullkw'] = keyword_label_test.fit_transform(testData['fullkw'].values)

print(trainData)
print(testData)


'''
# 将length分组并编码
def length_encode(length):
    if length <= 0: return 0
    elif length <= 10: return 1
    elif length <= 20: return 2
    elif length <= 40: return 3
    elif length <= 80: return 4
    elif length <= 120: return 5
    elif length <= 200: return 6


for dataset in full_data:
    #dataset['length'] = dataset.length.map(length_encode)
    dataset['length'] = dataset['length'].map(length_encode)
'''

print('\n'+'-'*10+'最后去掉一些不需要的列'+'-'*10)
drop_columns = ['id', 'keyword', 'location', 'text']
trainData.drop(drop_columns, axis=1, inplace=True)
testData.drop(drop_columns, axis=1, inplace=True)
print(testData)


print('\n'+'-'*10+'利用热力图来查看特征之间的相关关系(线性无关)'+'-'*10)
colormap = plt.cm.RdBu
plt.figure()
sns.heatmap(trainData.astype(float).corr(), cmap=colormap, linecolor='white', annot=True)
#plt.show()


print('\n'+'-'*10+'准备训练模型'+'-'*10)
X_train = trainData.iloc[:, 1:]
y_train = trainData['target']
X_test = testData
kf = KFold(n_splits=10, random_state=0)  # 分割10份数据，用于交叉验证


print('\n'+'-'*10+'训练逻辑回归模型'+'-'*10)
t0 = time.time()

C = [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6]
accuracy = dict()
for c in C:
    ls = LogisticRegression(penalty='l2', C=c, solver='lbfgs', random_state=0)
    cross_scores = cross_val_score(ls, X_train, y=y_train, cv=kf)
    accuracy[c] = np.mean(cross_scores)
print('best C:', sorted(accuracy.items(), key=lambda x:x[1], reverse=True)[0])

lr = LogisticRegression(penalty='l2', C=1, solver='lbfgs', random_state=0)
lr.fit(X_train, y_train)
y_test = lr.predict(X_test)
df = pd.DataFrame({'target': y_test})
df.to_csv(pwd+'predict_LR.csv', index=True)

t1 = time.time()
print('time cost:', t1 - t0, 's')


'''
print('\n'+'-'*10+'训练SVM'+'-'*10)
t0 = time.time()
svc = SVC(random_state=0)
params = {'kernel': ['linear', 'rbf', 'sigmoid'],\
        'C': [1, 1.2, 1.4, 1.5, 1.8, 2, 2.5, 3, 4, 10, 20, 50],\
        'gamma': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]}
best_model_svc = GridSearchCV(svc, param_grid=params, refit=True, cv=kf).fit(X_train, y_train)
print('best accuracy', best_model_svc.best_score_)
print('best parameters', best_model_svc.best_params_)
t1 = time.time()
print('time cost:', t1 - t0, 's')



print('\n'+'-'*10+'训练随机森林'+'-'*10)

t0 = time.time()

params = {'n_estimators': [50, 100, 150, 200, 250], \
          'max_depth': [3, 5, 7], 'min_samples_leaf': [2, 4, 6]}
RF = RandomForestClassifier(random_state=0)
best_model_rf = GridSearchCV(RF, param_grid=params, refit=True, cv=kf).fit(X_train,y_train)
print('best accuracy', best_model_rf.best_score_)
print('best parameters', best_model_rf.best_params_)

rf = RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_leaf=2, random_state=0)
rf.fit(X_train, y_train)
t1 = time.time()
print('time cost:', t1 - t0, 's')

y_test_pred = rf.predict(X_test)
df = pd.DataFrame({'target': y_test_pred})
df.to_csv(pwd+'predict_RF.csv', index=True)


print('\n'+'-'*10+'训练GBDT'+'-'*10)
t0 = time.time()
gbdt = GradientBoostingClassifier(random_state=0)
params ={'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2], \
         'n_estimators': [100, 300, 500], 'max_depth': [3, 5, 7]}
best_model_gbdt = GridSearchCV(gbdt, param_grid=params, refit=True).fit(X_train, y_train)
print('best accuracy', best_model_gbdt.best_score_)
print('best parameters', best_model_gbdt.best_params_)
t1 = time.time()
print('time cost:', t1 - t0, 's')
'''
'''
print('\n'+'-'*10+'训练XGBC'+'-'*10)
t0 = time.time()
xbc = XGBClassifier(random_state=0)
params = {'n_estimators': [50, 100, 300, 500], \
         'max_depth': [2, 3, 5, 7], 'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.25]}
best_model_xbc = GridSearchCV(xbc, param_grid=params, refit=True, cv=kf).fit(X_train, y_train)
print('best accuracy', best_model_xbc.best_score_)
print('best parameters', best_model_xbc.best_params_)
t1 = time.time()
print('time cost:', t1 - t0, 's')

'''
'''
print('\n'+'-'*10+'训练集成学习模型'+'-'*10)
t0 = time.time()
models = [('lr':LogisticRegression(penalty='l2', C=2, solver='lbfgs', random_state=0)), \
        ('svc':SVC(kernel='rbf', C=1.2, gamma=0.1, random_state=0)), \
        ('rf':RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=2, random_state=0)), \
        ('gbdt':GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, max_depth=3, random_state=0)), \
        ('xgbc':XGBClassifier(learning_rate=0.01, max_depth=2, n_estimators=300, random_state=0))]

vote = VotingClassifier(models, voting='hard')
vote.fit(X_train, y_train)

t1 = time.time()
print('time cost:', t1 - t0, 's')

print('\n'+'-'*10+'预测测试集'+'-'*10)
t0 = time.time()
y_test_pred = vote.predict(X_test)
t1 = time.time()
print('time cost:', t1 - t0, 's')
'''
