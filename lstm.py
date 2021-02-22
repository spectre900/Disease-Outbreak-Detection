import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import RNN

EPOCHS = 10
MAX_LEN = 100
BATCH_SIZE = 128
TRAIN_RATIO = 0.9

def getData():
    tweets = np.array([])
    category = np.array([])
    return tweets, category

def getDict():
    wordToNum = {}
    numToVector = {}
    return wordToNum,numToVector

def tokenize(tweets,wordToNum):
    
    tokenizedTweets = np.zeros((len(tweets),MAX_LEN),dtype='int')
    
    for i in range(len(tweets)):
        tweet = tweets[i]
        tweet = tweet.split()
        for j in range(len(tweet)):
            tokenizedTweets[i][j] = wordToNum[tweet[j]]
        
    return tokenizedTweets

def loadEmbdMatrix(numToVector):

    dModel = len(numToVector[0])
    vocabSize = len(numToVector)
    
    embdMatrix = np.zeros((vocabSize,dModel))
    
    for i in range(vocabSize):
        embdMatrix[i] = numToVector[i]
        
    return embdMatrix

def getModel(embdMatrix): 

    dModel = len(embdMatrix[0])
    vocabSize = len(embdMatrix)

    model = Sequential()
    model.add(Embedding(vocabSize,dModel,weights=[embdMatrix],input_length=MAX_LEN,trainable=True,mask_zero=True))
    model.add(LSTM(dModel))
    model.add(Dense(dModel, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-5, l2=2e-4),bias_regularizer=regularizers.l2(2e-4),activity_regularizer=regularizers.l2(2e-5)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    
    return model

def trainModel():
    tweets, category       = getData()
    wordToNum, numToVector = getDict()

    tweets      = tokenize(tweets,wordToNum)
    embdMatrix  = loadEmbdMatrix(numToVector)
    model       = getModel(embdMatrix)
    
    trainLength = (int)(len(tweets)*TRAIN_RATIO)
    xTrain      = tweets[:trainLength,:]
    xTest       = tweets[trainLength:,:]
    yTrain      = category[:trainLength:,:]
    yTest       = category[trainLength:,:]
    
    history     = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    return model,history

def plot(history):
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy and loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

model,history = trainModel()
plot(history)