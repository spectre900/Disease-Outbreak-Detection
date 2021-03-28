import twitterScraper as ts
from gensim.models import KeyedVectors
import pandas as pd
import re
import numpy as np

def getVocab(filename,d_model): # to generate word-num-vector mapping using pre trained word vectors
    
    file = open(filename,'r',encoding="utf8")
    
    num     = 0
    vocab_num_to_vector = {}
    vocab_word_to_num   = {}
    vocab_num_to_vector[num] = np.zeros((d_model))
    vocab_word_to_num['ukn'] = num
    num+=1

    for lines in file:
        values = lines.split()
        word   = values[0]
        vector = list(map(float,values[1:]))
        vocab_num_to_vector[num] = np.array(vector)
        vocab_word_to_num[word]  = num
        num+=1
    
    return vocab_word_to_num,vocab_num_to_vector


def tokenize(reviews,vocab_word_to_num,max_len): # tokenizing the tweets
    
    tokenized_reviews = np.zeros((len(reviews),max_len),dtype='int')
    
    for i in range(len(reviews)):
        review = reviews[i]
        review = review.split()
        for j in range(len(review)):
            tokenized_reviews[i][j] = vocab_word_to_num[review[j]]    
        
    return tokenized_reviews


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"gonna", "going to", phrase)
    phrase = re.sub(r"wanna", "want to", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\.+", " ", phrase)#dots (...)
    phrase = re.sub(r"[^A-Za-z]", " ", phrase)
    phrase = re.sub(r" +", " ", phrase)#spaces
    
    #emojis removing
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    emoji_pattern.sub(r'', phrase)
    
    return phrase

#store raw tweets in dataframe
count   = 100
keyword = 'dengue outbreak'
df=ts.TwitterSearchScraper(keyword,count).getSearchDataFrame()

#generate vocab
print('generating vocab')
vocab_word_to_num,vocab_num_to_vector = getVocab('glove.twitter.27B.100d.txt',100)
tweets=df['content'].values

#cleaning of tweets
print('cleaning tweets...')
processed_tweets=[]
for i in range(len(tweets)):
    print('tweet',i+1)
    tweet=tweets[i]
    words=tweet.split()
    processed_words=[]
    for j in range(len(words)):
        if words[j][0]!='@' and words[j][0:4]!="http":
            words[j]=words[j].lower()
            processed_words.append(words[j])
    tweet=' '.join(processed_words)
    tweet=decontracted(tweet)
    #print(tweet)
    processed_tweets.append(tweet)

#replacing unknown words
print('replacing unknowns...')
processed_tweets_final=[]
for i in range(len(processed_tweets)):
    print('tweet',i+1)
    tweet=processed_tweets[i]
    words=tweet.split()
    processed_words=[]
    for word in words:
        if word not in vocab_word_to_num.keys():
            processed_words.append('ukn')
        else:
            processed_words.append(word)
    tweet=' '.join(processed_words)
    #print(tweet)
    processed_tweets_final.append(tweet)

#tokenizing tweets
print('tokeninzing...')
tokenized_tweets=tokenize(processed_tweets_final,vocab_word_to_num,100)

#Uncomment below line to see tokenized tweets
#print(tokenized_tweets)
print('done')
df1 = pd.DataFrame(list(zip(processed_tweets_final, tokenized_tweets)), columns =['ProceesedTweets', 'TokenizedTweets'])
print(df1)



