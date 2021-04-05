import twitterScraper as ts

count   = 4000
keyword = 'dengue outbreak'

df = ts.TwitterSearchScraper(keyword,count).getSearchDataFrame()
df.to_csv('data/data.csv')
