import lightgbm as lgb 
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import re
import pickle


#正答率保存用のデータフレーム
save = pd.DataFrame(index=[])
save_folder = r"C:\Users\takuya\Desktop\趣味_ツール\顔認識"
#人有データを開く
exist = pd.read_csv("人有.csv", encoding="shift-jis")

#人無データを開く
pres = pd.read_csv("人無.csv", encoding="shift-jis")

#人有無情報追加
exist['人有無'] = 1
pres['人有無'] = 0

data = pd.concat([exist,pres])
data = data.reset_index(drop=True)

data_y = data['人有無']
#data_y = data_y.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

data_x = data.drop('人有無', axis=1)

data_x = data_x.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

print(data_x)
#トレーニングデータから、検証データを分割
train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.29, random_state=0)

test_x, valid_x, test_y, valid_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=0)
test_x = test_x.reset_index(drop=True)
test_y = test_y.reset_index(drop=True)
print(train_x.shape[1])

#LightGBM用のデータセットに変換
lgb_train = lgb.Dataset(train_x, train_y)
lgb_eval = lgb.Dataset(valid_x, valid_y)

gbm = lgb.LGBMClassifier() # モデルのインスタンスの作成
gbm.fit(train_x, train_y) # モデルの学習

model_file = 'LightGBM_trained_model.pkl'
pickle.dump(gbm, open(model_file, 'wb'))

#予測
predicted = gbm.predict(test_x)
print(predicted)

#正答率を計算
j=0
count=0
for i in predicted:
    if test_y[j] == i:
        count+=1
    j+=1

#正答率を表示、保存
print(count/(len(test_y)))
save.loc[0,"正答率"] = count/(len(test_y))

#csvファイルに予測と実測の書き出し
sample = pd.DataFrame(index=[])
sample2 = pd.DataFrame(index=[])
sample['予測'] = predicted
sample2['実測'] = test_y
sample = pd.concat([sample,sample2],axis=1)
os.chdir(save_folder)
sample.to_csv('prediction_result.csv', encoding="shift-jis")

#グラフの保存
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot() 
ax.plot(test_y, label="true")
ax.plot(predicted, label="pred")
plt.legend()
#plt.show()
os.chdir(save_folder)
plt.savefig('正答グラフ.png')

#正答率をcsvに保存
os.chdir(save_folder)
save.to_csv('out_正答率.csv', encoding="shift-jis")