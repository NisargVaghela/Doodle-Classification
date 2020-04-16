#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from quickdraw import QuickDrawData
import keras
from keras.models import Sequential
from keras.layers import Input, Dense,Conv1D,MaxPooling1D,LSTM,TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.utils.data_utils import Sequence
from keras.utils.np_utils import to_categorical   
from sklearn import preprocessing
import pickle 
import matplotlib.pyplot as plt


# In[3]:


qd = QuickDrawData()


# In[4]:


from quickdraw import QuickDrawDataGroup
anvils = QuickDrawDataGroup("pizza")
print(anvils.drawing_count)
print(anvils.get_drawing(1).image_data)
print(anvils.get_drawing(1).strokes)
print(anvils.get_drawing(1).no_of_strokes)


# In[5]:



def draw_img(img):
    image = Image.new("RGB", (255,255), color=(255,255,255))
    drawing = ImageDraw.Draw(image)
    for stroke in img:
        print(stroke)
        for coordinate in range(len(stroke[0])-1):
            x1 = stroke[0][coordinate]
            y1 = stroke[1][coordinate]
            x2 = stroke[0][coordinate+1]
            y2 = stroke[1][coordinate+1]
            drawing.line((x1,y1,x2,y2), fill=(0,0,0), width=2)
    image.show()


# In[6]:


def get_data(lables):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for lable in lables:
        imgs = QuickDrawDataGroup(str(lable))
        for i in range(imgs.drawing_count):
            img = imgs.get_drawing(i)
            print(img.image_data)
            print("---------")
            print(img.name)
            if i<800:
                x_train.append(img.image_data)
                y_train.append(img.name)
            else:
                x_test.append(img.image_data)
                y_test.append(img.name)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return x_train ,y_train ,x_test ,y_test;

def normalize_data(x,y,le):
    temp_y=[]
    for i in range(len(x)):
        temp = []
        dx=[]
        dy=[]
        dt=[]
        dl=[]
        for j in range(len(x[i])): 
#             scaler = MinMaxScaler()
#             norm = np.array(x[i][j])            
#             scaler.fit(norm.T)
#             norm = np.array(scaler.transform(norm.T))
#             x[i][j] = norm.T
            for k in range(len(x[i][j][0])-1):
                x1 = x[i][j][0][k]
                y1 = x[i][j][1][k]
                x2 = x[i][j][0][k+1]
                y2 = x[i][j][1][k+1]
                del_x = x2-x1
                del_y = y2-y1
#                 enc = le.transform([y[i]])
#                 enc = to_categorical(enc, num_classes=len(drawing_lable))
#                 dl.append(enc[0])
                
                if k==len(x[i][j][0])-2:
                    dx.append(del_x)
                    dy.append(del_y)
                    dt.append(1)

                else:
                    dx.append(del_x)
                    dy.append(del_y)
                    dt.append(0)        
                    
                    
        temp = np.array([dx,dy,dt]); 
#         print(dl)
        enc = le.transform([y[i]])
        enc = to_categorical(enc, num_classes=len(drawing_lable))
        dl.append(enc[0])
        temp_y.append(np.array(dl));
        x[i]= np.array([temp.T]);

#         x[i] = np.array(x[i])
    return x,np.array(temp_y);


# In[64]:


drawing_lable = ['hat','pizza','cat','dolphin','cell phone']

le = preprocessing.LabelEncoder()
le.fit(drawing_lable)
onehot_drawing_lable = le.transform(drawing_lable)
print(onehot_drawing_lable)
onehot_drawing_lable = to_categorical(onehot_drawing_lable, num_classes=len(drawing_lable))
print(onehot_drawing_lable)

x_train , y_train,x_test , y_test = get_data(drawing_lable)


# In[65]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[66]:


# draw imgs
draw_img(x_train[2990])


# In[67]:


# print((x[0][0]))
# le.transform(['hat'])
# dsfsd= to_categorical(le.transform(['hat']), num_classes=len(drawing_lable))
# print(dsfsd[0])


# In[68]:


print(x_train[0][0])


# In[69]:


x_train,y_train = normalize_data(x_train,y_train,le) 


# In[70]:


x_test,y_test = normalize_data(x_test,y_test,le) 


# In[71]:


print(x_train[0][0])


# In[72]:


print(x_test[0].shape)


# In[73]:


print(y_train[0][0])


# In[74]:


class DataGenerator(Sequence):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, index):    
        return self.X[index], np.array([self.Y[index][0]])


# In[75]:


train = DataGenerator(x_train,y_train) 


# In[76]:


a,b = train.__getitem__(200)
print(a.shape)
print(b.shape)


# In[77]:


def get_model():
    model = Sequential()
    model.add(Conv1D(32,2,input_shape=(None,3), activation='sigmoid'))    
#     model.add(Conv1D(64,2,activation='sigmoid'))
#     model.add(Conv1D(128,3,activation='sigmoid'))
    model.add(LSTM(64,activation='tanh',recurrent_activation='sigmoid', use_bias=True,return_sequences=True))
    model.add(LSTM(64,activation='tanh', recurrent_activation='sigmoid', use_bias=True))    
    model.add(Dense(len(drawing_lable),activation='softmax'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
    


# In[78]:


model = get_model()


# In[79]:


quality = model.fit_generator(DataGenerator(x_train,y_train),validation_data=DataGenerator(x_test,y_test),validation_freq=1, epochs=20, verbose=1,shuffle=True)


# In[80]:


ind = 500
pred = model.predict(x_test[ind])
# pred = np.mean(pred[0], axis=0)
print(pred)
print(y_test[ind][0])
result = np.where(pred[0] == np.amax(pred[0]))
le.inverse_transform(result)


# In[81]:


class Model:  
  
  def __init__(self,le,model):
    self.le = le
    self.model = model
  


# In[82]:


model = Model(le=le,model=model)


# In[83]:


pickle.dump(model, open('model.pkl', 'wb'))


# In[84]:


model  =  pickle.load(open('model.pkl', 'rb'))


# In[85]:


model.model.evaluate_generator(DataGenerator(x_test,y_test),verbose=1)


# In[86]:


ans = model.model.predict(x_test[200])
# x_test[200].shape
ans


# In[87]:


plt.plot(quality.history['acc'])
plt.plot(quality.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(quality.history['loss'])
plt.plot(quality.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[131]:


print(quality.history)


# In[ ]:




