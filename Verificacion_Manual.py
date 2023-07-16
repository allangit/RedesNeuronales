import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from scipy.spatial.tests.test_distance import metric
from nltk.sentiment.util import split_train_test




class manual:
    
    def __init__(self):
        
        self.df=pd.read_csv("diabetes.csv")
        self.x=self.df.iloc[:,0:8]
        self.y=self.df.iloc[:,8]
        self.model=0
        self.prediccion=0
        self.accuracy=0
        self.xtrain=0
        self.xtest=0
        self.ytrain=0
        self.ytest=0
        
        
    def split_data(self):
        
        self.xtrain,self.xtest, self.ytrain,self.ytest=train_test_split(self.x,self.y,test_size=0.33)

          
    def build_red(self):
        
        self.model=Sequential()
        self.model.add(Dense(12,input_dim=8,activation='relu'))
        self.model.add(Dense(80,activation='relu'))
        #self.model.add(Dense(50,activation='relu'))
        self.model.add(Dense(1,activation='sigmoid'))
        
    def compiler(self):
        
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        
    def ajustar_modelo(self):
    
        self.model.fit(self.xtrain,self.ytrain,validation_data=(self.xtest,self.ytest),epochs=170,batch_size=10)
        
        

   
        
        
        
        
        
        
        
        