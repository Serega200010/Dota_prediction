import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import time
import datetime
from sklearn.preprocessing import StandardScaler
N = 108
length = 97230

kf = KFold(shuffle = True,random_state=42,n_splits = 5)


#Загрузка и масштабирование признаков
features = pd.read_csv('features.csv', index_col='match_id').fillna(0)
features_for_train = features[features.columns[:-6]].drop(['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero'],1)

scaler = StandardScaler()
scaler.fit(features_for_train)
features_for_train = scaler.transform(features_for_train)
targets = features['radiant_win'].to_numpy()


#Поиск уникальных идентификаторов героев, и составление мешка слов по ним
Heroes_names1 = [set(features['r'+str(i)+'_hero'].value_counts().index) for i in range(1,6)]
Heroes_names2 = [set(features['d'+str(i)+'_hero'].value_counts().index) for i in range(1,6)]
Heroes_names = Heroes_names2 + Heroes_names1
HN = []

for S in Heroes_names:
    for h in S:
        HN.append(h)
HN = list(set(HN))
Nums = {HN[i] : i for i in range(len(HN))}
X_pick = np.zeros((features.shape[0], N))

for i,m_ind in enumerate(features.index):
    for p in range(5):
         num_r = features['r'+str(p+1)+'_hero'][m_ind]
         num_d = features['d'+str(p+1)+'_hero'][m_ind]
         X_pick[i][Nums[num_r]] = 1
         X_pick[i][Nums[num_d]] = -1

features_for_train = np.concatenate((features_for_train,np.array(X_pick)),axis = 1)
print('Shapes of f_f_t: ', features_for_train.shape)
X_train, x_test, y_train, y_test = train_test_split(features_for_train, targets,test_size=0.8,random_state=241)



#Cross_validation при разных значениях С и запись результатов в словарь scores
start_time = datetime.datetime.now()
scores = {}
c = [10]#[0.9,1,10,100]

for C1 in c:
     print('Start training with C = {}'.format(C1))
     clf = LogisticRegression(penalty = 'l2',C = C1,tol = 1e-5,random_state = 0,max_iter = 10000)
     clf.fit(features_for_train,targets)
     sc = make_scorer(roc_auc_score)
     cr_val = (cross_val_score(clf,features_for_train,targets,cv=kf,scoring = sc)).mean()
     scores[C1] = cr_val
     pr = clf.predict_proba(x_test)
     print('Accuracy_score = ',clf.score(x_test,y_test))
     print('cr_val score = ',cr_val)
t = datetime.datetime.now() - start_time    


with open('report_lr.txt','a') as fp:
    fp.write('\n' + ':: '+ str(scores) + '\nEllapsed time = ' + str(t))




#Поиск лучшего значения С по словарю и инициализация логистической регрессии с найденным лучшим значением
m = max(scores.values())
C = 0
for i in scores.keys():
    if scores[i] == m:
        C = i

final_model = LogisticRegression(penalty = 'l2',C = C,tol = 1e-5,random_state = 0,max_iter = 10000)
final_model.fit(features_for_train,targets)



#Загрузка тестовой выборки и ее предобработка, включая формирование мешка слов и его конкатинации, аналогично с тренировочной выборкой
features_test = pd.read_csv('features_test.csv').fillna(0)
features_to_predict = features_test.drop(['match_id','lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero'],1)


X_pick_test = np.zeros((features_test.shape[0], N))
for i,m_ind in enumerate(features_test.index):
    for p in range(5):
         num_r = features_test['r'+str(p+1)+'_hero'][m_ind]
         num_d = features_test['d'+str(p+1)+'_hero'][m_ind]
         X_pick_test[i][Nums[num_r]] = 1
         X_pick_test[i][Nums[num_d]] = -1

scaler2 = StandardScaler()
scaler2.fit(features_to_predict)
features_to_predict = scaler2.transform(features_to_predict)
features_to_predict = np.concatenate((features_to_predict,np.array(X_pick_test)),axis = 1)





# Построение предсказаний по тестовой выборке
probs = final_model.predict_proba(features_to_predict)
r_win = [probs[i][0] for i in range(len(probs))]
answer = pd.DataFrame({'match_id' : features_test['match_id'].to_numpy(), 'radiant_win' : r_win})
answer.to_csv('result.csv')





