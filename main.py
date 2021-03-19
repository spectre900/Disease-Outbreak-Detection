import twitterScraper as ts

count   = 2000
keyword = 'dengue outbreak'

df = ts.TwitterSearchScraper(keyword,count).getSearchDataFrame()
df.to_csv('data.csv')
