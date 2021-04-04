import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import geopandas as gpd
from geotext import GeoText
from tensorflow import keras
from iso3166 import countries
import matplotlib.pyplot as plt
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
        
        if predictions[i][0] >= 0:
            
            dict = getLocation(tweets[i])
            for country in dict.keys():
                worldMap[country] += dict[country]
    
    return worldMap

def run():

    tweetsTokens, tweets = getData()
    model = getModel()
    predictions = model.predict(tweetsTokens)
    worldData = getCountryDict(tweets, predictions)
    worldMap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    worldMap['dengue'] = np.nan
    
    for i in range(worldMap.shape[0]):
        country_code_alpha3 = worldMap['iso_a3'][i]
        if country_code_alpha3 != '-99':
            country_code_alpha2 = countries.get(country_code_alpha3).alpha2
            worldMap['dengue'][i] = worldData[country_code_alpha2]
        else:
            worldMap['dengue'][i] = 0
    
    worldMap.plot(column='dengue')
    plt.show()

run()