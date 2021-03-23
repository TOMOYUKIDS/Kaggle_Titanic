import os

path = os.getcwd()

print(path)

os.chdir('../')

print(os.getcwd())


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
import lightgbm

train = pd.read_csv("C:/Users/tomoyuki.kawashita/Documents/Tomo/006.Kaggle/003.Titanic/data/train.csv")
test = pd.read_csv("C:/Users/tomoyuki.kawashita/Documents/Tomo/006.Kaggle/003.Titanic/data/test.csv")

print('train.shape:',train.shape)
print('test.shape:',test.shape)

alldata=pd.concat([train,test],axis=0)
print("alldata.shape:",alldata.shape)



#特徴量と欠損値の割合を確認
for col in alldata.columns:
    print(col,round(alldata[col].isna().sum()/alldata.shape[0]*100,2))


alldata = alldata.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)

for col in alldata.columns:
    print(col,alldata[col].isna().sum())

alldata.describe(include='all')

alldata[["Age"]]=alldata[["Age"]].fillna(alldata[["Age"]].mean())
alldata[["Fare"]]=alldata[["Fare"]].fillna(alldata[["Fare"]].mean())
alldata[["Embarked"]]=alldata[["Embarked"]].fillna("S")

alldata=pd.get_dummies(alldata)

sc = StandardScaler()
alldata[["Age"]]=sc.fit_transform(alldata[["Age"]])
alldata[["Fare"]]=sc.fit_transform(alldata[["Fare"]])

target_col = 'Survived'
feature_cols = [col for col in alldata.columns if col not in target_col]

X = alldata[feature_cols]
y = alldata[target_col]

X_train = X.iloc[:train.shape[0],]
X_test = X.iloc[train.shape[0]:,]
y_train = y.iloc[:train.shape[0],]
y_test = y.iloc[test.shape[0]:,]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

LGBMC = lightgbm.LGBMClassifier()
LGBMC.fit(X_train1, y_train1)

LGBMC.score(X_test1, y_test1)

y_pred=LGBMC.predict(X_test1)

scores=cross_val_score(LGBMC,X_train1, y_train1, cv=5, n_jobs=-1)
print('Score mean:{0} std:{1}'.format(round(np.mean(scores),2),round(np.std(scores),2)))

pred = LGBMC.predict(X_test)

dt=pd.DataFrame(test['PassengerId'],pred)
dt.columns 

print('############')

def greeting():
    yield 'Good morning'
    yield 'Good afternoon'
    yield 'Good night'

g = greeting()
print(next(g))
print('@@@@@')
print(next(g))
print('@@@@@')
print(next(g))
