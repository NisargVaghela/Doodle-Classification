import pickle

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from keras.models import Sequential
# from keras.layers import LSTM, Conv1D, Dense, Input, MaxPooling1D
from keras.utils import Sequence, to_categorical
from PIL import Image, ImageDraw
# from quickdraw import QuickDrawData
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# from tensorflow.keras.models import Sequential

app = Flask(__name__,template_folder='pages')

class Model:  
  
  def __init__(self,le,model):
    self.le = le
    self.model = model


def normalize_data(x,le):
    temp = []
    dx=[]
    dy=[]
    dt=[]        
    for i in range(len(x)):
        for k in range(len(x[i][0])-1):
            x1 = x[i][0][k]
            y1 = x[i][1][k]
            x2 = x[i][0][k+1]
            y2 = x[i][1][k+1]
            del_x = x2-x1
            del_y = y2-y1
            
            if k==len(x[i][0])-2:
                dx.append(del_x)
                dy.append(del_y)
                dt.append(1)

            else:
                dx.append(del_x)
                dy.append(del_y)
                dt.append(0)

    temp = np.array([np.array(dx),np.array(dy),np.array(dt)]);        
    
    return np.array([temp.T]);  


model  =  pickle.load(open('model.pkl', 'rb'))
# graph = tf.compat.v1.get_default_graph()
ans = []

@app.route('/')
def send_page():	
	return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict_activity():    
    content = request.get_json()    
    content = content['data']    
    # print(content)
    x = normalize_data(content,model.le)      
    x = np.array(x)
    # print(x)
    # with graph.as_default():
    pred = model.model.predict(x)
    print(pred[0])
    # pred = np.mean(pred[0], axis=0)
    # print(pred)        
    result = np.where(pred[0] == np.amax(pred[0]))
    print(result)
    ans = model.le.inverse_transform(result)
    print(ans[0])    
    return jsonify({'ans': ans[0]})

if __name__ == '__main__':
   app.run(debug = True)