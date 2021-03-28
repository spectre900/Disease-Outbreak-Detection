import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow.keras.backend as K

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

EPOCHS = 10
MAX_LEN = 100
LSTM_DIM = 100
DENSE_DIM = 50
BATCH_SIZE = 16
VECTOR_DIM = 50
VALID_SPLIT = 0.15

def getData():
    tweets = np.load('processed/tokenized_tweets.npy')
    category = pd.read_csv('data_annot.csv')['y'].values.reshape(-1,1)
    return tweets, category

def getDict():
    numToVector = pkl.load(open('processed/num_to_vec.pkl','rb'))
    return numToVector

def tokenToVec(tweets, numToVector):
    
    x = np.zeros((len(tweets), MAX_LEN, VECTOR_DIM))

    for i in range(len(tweets)):
        for j in range(MAX_LEN):
            x[i][j] = numToVector[tweets[i][j]]

    return x

def getModel(): 

    model = Sequential()
    model.add(Input(shape=(MAX_LEN, VECTOR_DIM)))
    model.add(LSTM(LSTM_DIM))
    model.add(Dense(DENSE_DIM))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def trainModel():
    tweets, category = getData()
    numToVector = getDict()

    model = getModel()

    print(model.summary())

    x = tokenToVec(tweets, numToVector)
    y = category
    
    history = model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALID_SPLIT)

trainModel()