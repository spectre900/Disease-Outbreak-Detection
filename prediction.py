import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
from geotext import GeoText
from tensorflow import keras
from collections import defaultdict


def getData():
    tweetsTokens = np.load('processed/tokenized_tweets_predict.npy')
    tweets = pkl.load(open('processed/geotext_tweets_predict.pkl','rb'))
    return tweetsTokens, tweets

def getModel():

   model = keras.models.load_model('lstm')
   return model

def getLocation(text):
    return GeoText(text).country_mentions

def getCountryDict(tweets, predictions):

    worldMap = defaultdict(int)

    for i in range(len(tweets)):
        
        if predictions[i][0] >= 0.4:
            
            dict = getLocation(tweets[i])
            for country in dict.keys():
                worldMap[country] += dict[country]
    
    return worldMap

def run():

    tweetsTokens, tweets = getData()
    model = getModel()
    predictions = model.predict(tweetsTokens)
    worldMap = getCountryDict(tweets, predictions)
    print(sorted(worldMap.items(), key=lambda x: x[1]))

run()