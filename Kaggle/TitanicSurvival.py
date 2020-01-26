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
#from xgboost import XGBClassifier

pwd = 'D:\\Python\\Titanic\\Data\\'
trainData = pd.read_csv(pwd+'train.csv')  # ����ѵ������
testData = pd.read_csv(pwd+'test.csv')  # ���ز�������

full_data = [trainData, testData]
trainData.info()  # ѵ�����ݵ���Ϣ
testData.info()
trainData.sample(n=10)  # ����鿴ѵ������
testData.sample(n=10)

print('\n'+'-'*10+'ͳ������������'+'-'*10)
print(trainData["Survived"].value_counts(normalize=True))

print('\n'+'-'*10+'������ɢ�����������ʵĹ�ϵ'+'-'*10)
f, axes = plt.subplots(2, 3, sharey=True)
sns.factorplot(x='Pclass', y='Survived', data=trainData, kind='bar', ax=axes[0, 0])
sns.barplot(x='Sex', y='Survived', data=trainData, ax=axes[0, 1])
sns.barplot(x='SibSp', y='Survived', data=trainData, ax=axes[0, 2])
sns.barplot(x='Parch', y='Survived', data=trainData, ax=axes[1, 0])
sns.barplot(x='Embarked', y='Survived', data=trainData, ax=axes[1, 1])
#plt.show()

print('\n'+'-'*10+'�������������������ʵĹ�ϵ'+'-'*10)
f, axes = plt.subplots(2, 1, sharey=True)
sns.kdeplot(trainData["Age"][(trainData["Survived"] == 0) & (trainData["Age"].notnull())], \
            color="Red", shade=True, ax=axes[0], label='Not Survived')
sns.kdeplot(trainData["Age"][(trainData["Survived"] == 1) & (trainData["Age"].notnull())], \
            color="Blue", shade=True, ax=axes[0], label='Survived')
sns.kdeplot(trainData["Fare"][(trainData["Survived"] == 0) & (trainData["Fare"].notnull())], \
            color="Red", shade=True, ax=axes[1], label='Not Survived')
sns.kdeplot(trainData["Fare"][(trainData["Survived"] == 1) & (trainData["Fare"].notnull())], \
            color="Blue", shade=True, ax=axes[1], label='Survived')
#plt.show()

print('\n'+'-'*10+'Pclass��Sex��Survived�Ĺ�ϵ'+'-'*10)
f, axes = plt.subplots(1, 1, sharey=True)
sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=trainData, ax=axes)
#plt.show()

print('\n'+'-'*10+'������ͼ������쳣ֵ����Ⱥ��'+'-'*10)
f, axes = plt.subplots(1, 2, sharey=True)
sns.boxplot(y='Age', data=trainData, ax=axes[0])
sns.boxplot(y='Fare', data=trainData, ax=axes[1])
#plt.show()

print('\n'+'-'*10+'ȱʧֵ��ȫ'+'-'*10)
# �����ν���ڳ��ֳ���title�У������滻ΪRare
def fix_title(title):
    if title.strip() not in ['Mr', 'Miss', 'Mrs', 'Master', 'Dr']:
        return 'Rare'
    else:
        return title.strip()

# �ú�����ͬtitle���˵�����ľ�ֵ��������ȱʧֵ
for dataset in full_data:
    # ��ȡ�����е�title
    dataset['Title'] = dataset['Name'].str.extract(', (\S+). ', expand=False)
    # ��һЩ���ִ������ٵ�title�滻ΪRare
    dataset['Title'] = dataset['Title'].map(fix_title)
    # �滻�����е�ȱʧֵ
    dataset['Age'] = dataset.groupby('Title')['Age'].apply(lambda x:x.fillna(x.mean()))

    # ��Embarked���������ȱʧֵ
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

    # ���������е�Fare��ȱʧֵ���������λ�������
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

# �鿴ѵ�����Ͳ��Լ��е�ȱʧֵ���
print(trainData.isnull().sum())
print(testData.isnull().sum())

print('\n'+'-'*10+'��������'+'-'*10)
def is_mother(row):
    if row['Title'] == 'Mrs' and row['Age'] > 18 and row['Parch'] > 0:
        return 1
    else:
        return 0

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = dataset.apply(lambda x:1 if x['FamilySize'] == 1 else 0, axis=1)
    dataset['Mother'] = dataset.apply(is_mother, axis=1)
    dataset['Child'] = dataset.apply(\
        lambda row:1 if row['Age'] < 18 and row['IsAlone'] != 1 else 0, axis=1)

print('\n'+'-'*10+'��������'+'-'*10)
# ���������ת��Ϊ��ֵ��
sex_mapping = {'female': 0, 'male': 1}
embarked_mapping = {'S': 1, 'C': 2, 'Q': 3}
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rare': 6}

for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    # ���ⴴ��һ��������Class_Sex
    dataset['Class_Sex'] = dataset['Pclass'] * dataset['Sex']
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
    dataset['Title'] = dataset['Title'].map(title_mapping)

# ��Age��Fare���鲢����
def age_encode(age):
    if age <= 18: return 0
    elif age <= 32: return 1
    elif age <= 48: return 2
    elif age <= 64: return 3
    elif age <= 80: return 4

def fare_encode(fare):
    if fare <= 7: return 0
    elif fare <= 14: return 1
    elif fare <= 31: return 2
    elif fare <= 600: return 3

for dataset in full_data:
    dataset['Age'] = dataset.Age.map(age_encode)
    dataset['Fare'] = dataset.Fare.map(fare_encode)

# ���ȥ��һЩ����Ҫ����
drop_columns = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']
trainData.drop(drop_columns, axis=1, inplace=True)
testData.drop(drop_columns, axis=1, inplace=True)

print('\n'+'-'*10+'��������ͼ���鿴����֮�����ع�ϵ(�����޹�)'+'-'*10)
colormap = plt.cm.RdBu
plt.figure()
sns.heatmap(trainData.astype(float).corr(), cmap=colormap, linecolor='white', annot=True)
plt.show()

print('\n'+'-'*10+'׼��ѵ��ģ��'+'-'*10)
X_train = trainData.iloc[:, 1:]
y_train = trainData['Survived']
X_test = testData
kf = KFold(n_splits=10, random_state=0)  # �ָ�10�����ݣ����ڽ�����֤

print('\n'+'-'*10+'ѵ���߼��ع�ģ��'+'-'*10)
t0 = time.time()
C = [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6]
accuracy = dict()
for c in C:
    ls = LogisticRegression(penalty='l2', C=c, solver='lbfgs', random_state=0)
    cross_scores = cross_val_score(ls, X_train, y=y_train, cv=kf)
    accuracy[c] = np.mean(cross_scores)
print('best C:', sorted(accuracy.items(), key=lambda x:x[1], reverse=True)[0])
t1 = time.time()
print('time cost:', t1 - t0, 's')

print('\n'+'-'*10+'ѵ��SVM'+'-'*10)
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


print('\n'+'-'*10+'ѵ�����ɭ��'+'-'*10)

t0 = time.time()
params = {'n_estimators': [50, 100, 150, 200, 250], \
          'max_depth': [3, 5, 7], 'min_samples_leaf': [2, 4, 6]}
RF = RandomForestClassifier(random_state=0)
best_model_rf = GridSearchCV(RF, param_grid=params, refit=True, cv=kf).fit(X_train,y_train)
print('best accuracy', best_model_rf.best_score_)
print('best parameters', best_model_rf.best_params_)
t1 = time.time()
print('time cost:', t1 - t0, 's')


print('\n'+'-'*10+'ѵ��GBDT'+'-'*10)
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
print('\n'+'-'*10+'ѵ��XGBC'+'-'*10)
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
print('\n'+'-'*10+'ѵ������ѧϰģ��'+'-'*10)
t0 = time.time()
models = [('lr':LogisticRegression(penalty='l2', C =2, solver='lbfgs', random_state=0)), \
        ('svc':SVC(kernel='rbf', C=1.2, gamma=0.1, random_state=0)), \
        ('rf':RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=2, random_state=0)), \
        ('gbdt':GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, max_depth=3, random_state=0)), \
        ('xgbc':XGBClassifier(learning_rate=0.01, max_depth=2, n_estimators=300, random_state=0))]

vote = VotingClassifier(models, voting='hard')
vote.fit(X_train, y_train)

t1 = time.time()
print('time cost:', t1 - t0, 's')

print('\n'+'-'*10+'Ԥ����Լ�'+'-'*10)
t0 = time.time()
y_test_pred = vote.predict(X_test)
t1 = time.time()
print('time cost:', t1 - t0, 's')
'''
