
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from keras.models import Sequential 
from keras.layers import Dense
from scipy.spatial.tests.test_distance import metric
from fontTools.misc.py23 import round


class modelado:
    
    def __init__(self):
        
        self.df=pd.read_csv("diabetes.csv")
        self.x=self.df.iloc[:,0:8]
        self.y=self.df.iloc[:,8]
        self.model=0
        self.prediccion=0
        self.accuracy=0
        
    def build_red(self):
        
        self.model=Sequential()
        self.model.add(Dense(12,input_dim=8,activation='relu'))
        self.model.add(Dense(80,activation='relu'))
        #self.model.add(Dense(50,activation='relu'))
        self.model.add(Dense(1,activation='sigmoid'))
        
    def compiler(self):
        
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        
    def ajustar_modelo(self):
    
        self.model.fit(self.x,self.y,epochs=270,batch_size=16)
        
    def evaluar_modelo(self):
        
        _,self.accuracy=self.model.evaluate(self.x,self.y)
        print("El promedio es {}:::".format(self.accuracy*100))

    def predicciones(self):
        
        self.prediccion=self.model.predict(self.x)
        for i in range(10):
            
            print("prediccion {} -----> esperado {}".format(np.round(self.prediccion[i]),self.y[i]))

    
          
 

