import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization

EPOCHS = 10
MAX_LEN = 100
LSTM_DIM = 100
DENSE_DIM = 50
BATCH_SIZE = 16
VECTOR_DIM = 50
VALID_SPLIT = 0.3

def getData():
    tweets = np.load('processed/tokenized_tweets.npy')
    category = pd.read_csv('data_annot.csv')['y'].values.reshape(-1,1)
    return tweets, category

def getDict():
    numToVector = pkl.load(open('processed/num_to_vec.pkl','rb'))
    return numToVector

def getEmbedMatrix(numToVector):

    embed_matrix = np.zeros((len(numToVector), VECTOR_DIM))

    for i in range(len(numToVector)):

        embed_matrix[i] = numToVector[i]

    return embed_matrix

def getModel(embed_matrix): 

    model = Sequential()
    model.add(Input(shape=(MAX_LEN)))
    model.add(Embedding(len(embed_matrix),
                        output_dim = VECTOR_DIM,
                        input_length = MAX_LEN,
                        mask_zero = True,
                        trainable = False))
    model.add(LSTM(LSTM_DIM))
    model.add(Dense(DENSE_DIM))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def trainModel():

    tweets, category = getData()
    numToVector = getDict()

    embed_matrix = getEmbedMatrix(numToVector)

    model = getModel(embed_matrix)

    print(model.summary())

    x = tweets
    y = category

    print(x.shape, y.shape)
    
    history = model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALID_SPLIT)

trainModel()