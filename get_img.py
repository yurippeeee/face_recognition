import cv2
import numpy as np
import pandas as pd

save = pd.DataFrame(index=[])

# VideoCapture オブジェクトを取得します
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5)
cap.set(cv2.CAP_PROP_FPS, 30)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps:{}　width:{}　height:{}".format(fps, width, height))

col = []
for j in range(400):
    col.append("画素"+str(j))

i=0
while i < 700:
    ret, img = cap.read()
    img = cv2.resize(img , dsize=(20, 20))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # RGB2〜 でなく BGR2〜 を指定
    img = img.reshape(400)
    tmp = pd.DataFrame(data=img,index=col)
    tmp = tmp.transpose()
    save = pd.concat([save,tmp])
    
    cv2.imshow('frame',img)
    i+=1
save.to_csv('人無.csv', index=False, encoding="shift-jis")
cap.release()
cv2.destroyAllWindows()