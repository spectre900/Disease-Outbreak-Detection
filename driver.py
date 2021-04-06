import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import re
import os
import pygal
import cairosvg
import numpy as np
import pandas as pd
import pickle as pkl
import zipfile as unzip
from geotext import GeoText
from tensorflow import keras
from collections import defaultdict

from Scraper import twitterScraper as ts

from LSTM.train.preprocessing import getVocab
from LSTM.train.preprocessing import tokenize
from LSTM.train.preprocessing import decontracted
from LSTM.train.preprocessing import removeUnknowns
from LSTM.train.preprocessing import cleanTweets

#unzip
print('Unzipping Word2Vec...')
#unzip.ZipFile('LSTM/data/Word2Vec/glove6b50dtxt.zip', 'r').extractall('LSTM/data/Word2Vec/')

#read CSV
print('Getting Tweets...')
df = ts.TwitterSearchScraper('dengue outbreak', 500).getSearchDataFrame()

#generate vocab
print('generating vocab...')
vocab_word_to_num,vocab_num_to_vector = getVocab('LSTM/data/Word2Vec/glove.6B.50d.txt',50)
tweets=df['content'].values

#cleaning of tweets
print('cleaning tweets...')
processed_tweets_predict, geotext_tweets_predict = cleanTweets(tweets)

#replacing unknown words
print('replacing unknowns...')
processed_tweets_final_predict = removeUnknowns(processed_tweets_predict, vocab_word_to_num)

#tokenizing tweets
print('tokeninzing...')
tokenized_tweets_predict=tokenize(processed_tweets_final_predict,vocab_word_to_num,100)

#rename
tweetsTokens, tweets = tokenized_tweets_predict, geotext_tweets_predict

#load model
print('loading model...')
model = keras.models.load_model('LSTM/data/model.h5')

#classify tweets using lstm
print('classifying tweets...')
predictions = model.predict(tweetsTokens)

print('extracting mentioned countries...')
worldData = defaultdict(int)

#Get positive tweet indices
index = np.array(np.where(predictions[:,0] >= 0.5)[0])

#Iterate over those tweets and get countries
for i in index:
    dict = GeoText(tweets[i]).country_mentions
    for country in dict.keys():
        worldData[country.lower()] += dict[country]

#plot it on a world map
print('plotting world map...')
worldMap = pygal.maps.world.World()
worldMap.add('Dengue Outbreak', worldData)

#make dir
try:
    os.makedirs('Frontend/static/map/')
except FileExistsError:
    pass

#save the plot in svg format
worldMap.render_to_file('Frontend/static/map/map.svg')
cairosvg.svg2svg(url='Frontend/static/map/map.svg', write_to='Frontend/static/map/map.svg')

print('done')