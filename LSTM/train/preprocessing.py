import re
import os
import numpy as np
import pandas as pd
import pickle as pkl
import zipfile as unzip

# to generate word-num-vector mapping using pre trained word vectors
def getVocab(filename,d_model): 
    
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

# tokenizing the tweets
def tokenize(reviews,vocab_word_to_num,max_len):
    
    tokenized_reviews = np.zeros((len(reviews),max_len),dtype='int')
    
    for i in range(len(reviews)):
        review = reviews[i]
        review = review.split()
        for j in range(len(review)):
            tokenized_reviews[i][j] = vocab_word_to_num[review[j]]    
        
    return np.array(tokenized_reviews)


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

def cleanTweets(tweets):
    
    processed_tweets_final=[]
    geotext_tweets_final=[]
    for i in range(len(tweets)):
        tweet=tweets[i]
        words=tweet.split()
        processed_words=[]
        processed_words_geotext=[]
        for j in range(len(words)):
            if words[j][0]!='@' and words[j][0:4]!="http":
                processed_words_geotext.append(words[j])
                words[j]=words[j].lower()
                processed_words.append(words[j])
        geotext_tweet =' '.join(processed_words_geotext)
        geotext_tweet=decontracted(geotext_tweet)
        tweet=' '.join(processed_words)
        tweet=decontracted(tweet)
        geotext_tweets_final.append(geotext_tweet)
        processed_tweets_final.append(tweet)

    return processed_tweets_final, geotext_tweets_final

def removeUnknowns(processed_tweets, vocab_word_to_num):

    processed_tweets_final=[]
    for i in range(len(processed_tweets)):
        tweet=processed_tweets[i]
        words=tweet.split()
        processed_words=[]
        for word in words:
            if word not in vocab_word_to_num.keys():
                processed_words.append('ukn')
            else:
                processed_words.append(word)
        tweet=' '.join(processed_words)
        processed_tweets_final.append(tweet)

    return processed_tweets_final

def run():
	
    #Unzip Word2Vec
    print('Unzipping Word2Vec...')
    unzip.ZipFile('../data/Word2Vec/glove6b50dtxt.zip', 'r').extractall('../data/Word2Vec')

    #read CSV
    print('Reading CSV...')
    df = pd.read_csv('../../Scraper/data/data_annotated.csv')

    #generate vocab
    print('generating vocab...')
    vocab_word_to_num,vocab_num_to_vector = getVocab('../data/Word2Vec/glove.6B.50d.txt',50)
    tweets=df['content'].values

    #cleaning of tweets
    print('cleaning tweets...')
    processed_tweets, _ = cleanTweets(tweets)

    #replacing unknown words
    print('replacing unknowns...')
    processed_tweets_final=removeUnknowns(processed_tweets, vocab_word_to_num)

    #tokenizing tweets
    print('tokeninzing...')
    tokenized_tweets=tokenize(processed_tweets_final,vocab_word_to_num,100)

    try:
        os.makedirs('../data/processed/')
    except FileExistsError:
        pass

    print('Saving...')
    np.save('../data/processed/tokenized_tweets.npy', tokenized_tweets)
    pkl.dump(vocab_num_to_vector,open('../data/processed/num_to_vec.pkl','wb'))

if __name__ == '__main__':

    run()