import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import time
import datetime



#Загрузка признаков и удаление, позволяющих 'заглянуть в будущее'
features = pd.read_csv('features.csv', index_col='match_id')
features_for_train = features[features.columns[:-6]]
length = 97230



#Поиск стобцов с пропусками и их вывод на экран
for col in features_for_train.columns:
    if features_for_train[col].count()!=length:
        print('column {} contain only {}/{} fields'.format(col,features_for_train[col].count(),length))



#Дообаботка данных
features_for_train = features_for_train.fillna(0)
kf = KFold(shuffle = True,random_state=42,n_splits = 5)
targets = features['radiant_win'].to_numpy()
lr = 0.5
start_time = datetime.datetime.now()
X_train, x_test, y_train, y_test = train_test_split(features_for_train.to_numpy(), targets,test_size=0.6,random_state=241)



#Проведение кросс-валидации по набору значений количества деревьев из массива nums, сохранение результатов в словарь scores
scores = {}
nums = [25,30,35,40]
for num_of_trees in nums:#range(5,35,5):
     print(num_of_trees)
     clf  = GradientBoostingClassifier(n_estimators=num_of_trees, verbose=True, learning_rate = lr, random_state=241)
     sc = make_scorer(roc_auc_score)
     cr_val = (cross_val_score(clf,X_train,y_train,cv=kf,scoring = sc)).mean()
     scores[num_of_trees] = cr_val
print(scores)
print ('Time elapsed:', datetime.datetime.now() - start_time)
t = datetime.datetime.now() - start_time


#Запись результатов в файл 
with open('report.txt','a') as fp:
    fp.write('\nlr = ' + str(lr) + ': '+ str(scores) + '\nEllapsed time = ' + str(t))



