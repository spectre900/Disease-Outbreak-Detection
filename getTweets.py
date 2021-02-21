import twitterScraper as ts

count   = 100
keyword = 'dengue outbreak'

print(ts.TwitterSearchScraper(keyword,count).getSearchDataFrame())