import re
import time
import random
import pandas
import requests
import datetime
import email.utils
import urllib.parse

class Tweet():

	def __init__(self,content,dateTime):
		self.content = content
		self.dateTime= dateTime

class Scraper:

	def __init__(self):
		self.session = requests.Session()

	def request(self, method, url, params = None, headers = None):
		req = self.session.prepare_request(requests.Request(method, url, params = params, headers = headers))
		req = self.session.send(req)
		return req

	def get(self, *args, **kwargs):
		return self.request('GET', *args, **kwargs)

class TwitterAPIScraper(Scraper):
	def __init__(self, baseUrl):
		super().__init__()
		self.baseUrl = baseUrl
		self.guestToken = None
		self.authorizationHeader = 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs=1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'
		self.userAgent = f'Chrome/79.0.3945.{random.randint(0, 9999)}'
		self.apiHeaders = {
			'Referer': self.baseUrl,
			'User-Agent': self.userAgent,
			'Authorization': self.authorizationHeader
		}

	def setGuestToken(self):
		if self.guestToken is not None:
			return
		r = self.get(self.baseUrl, headers = {'User-Agent': self.userAgent})
		self.guestToken = re.search(r'document\.cookie = decodeURIComponent\("gt=(\d+); Max-Age=10800; Domain=\.twitter\.com; Path=/; Secure"\);', r.text).group(1)
		self.session.cookies.set('gt', self.guestToken, domain = '.twitter.com', path = '/', secure = True, expires = time.time() + 10800)
		self.apiHeaders['x-guest-token'] = self.guestToken

	def getApiData(self, endpoint, params):
		self.setGuestToken()
		r = self.get(endpoint, params = params, headers = self.apiHeaders)
		return r.json()

	def iterApiData(self, endpoint, params):
		while True:
			obj = self.getApiData(endpoint, params)
			yield obj
			for instruction in obj['timeline']['instructions']:
				if 'addEntries' in instruction:
					entries = instruction['addEntries']['entries']
				elif 'replaceEntry' in instruction:
					entries = [instruction['replaceEntry']['entry']]
				else:
					continue
				for entry in entries:
					if entry['entryId'] == 'sq-cursor-bottom':
						newCursor = entry['content']['operation']['cursor']['value']
			params['cursor'] = newCursor


	def instructionsToTweets(self, obj):
		for instruction in obj['timeline']['instructions']:
			if 'addEntries' in instruction:
				entries = instruction['addEntries']['entries']
			elif 'replaceEntry' in instruction:
				entries = [instruction['replaceEntry']['entry']]
			else:
				continue
			for entry in entries:
				if entry['entryId'].startswith('sq-I-t-'):
					tweet = obj['globalObjects']['tweets'][entry['content']['item']['content']['tweet']['id']]
					yield self.toTweet(tweet, obj)

	def toTweet(self, tweet, obj):
		content = re.sub('\\n',' ',tweet['full_text'])
		dateTime = email.utils.parsedate_to_datetime(tweet['created_at'])
		return Tweet(content,dateTime)


class TwitterSearchScraper(TwitterAPIScraper):

	def __init__(self, query, count):
		super().__init__(baseUrl = 'https://twitter.com/search?' + urllib.parse.urlencode({'f': 'live', 'lang': 'en', 'q': query, 'src': 'spelling_expansion_revert_click'}))
		self.query = query
		self.count = count

	def getItems(self):
		params = {
			'count': '100',
			'q': self.query,
			'tweet_mode': 'extended',
			'tweet_search_mode': 'live',
		}
		for obj in self.iterApiData('https://api.twitter.com/2/search/adaptive.json', params):
			yield from self.instructionsToTweets(obj)

	def getSearchDataFrame(self):
		tweetGenerator = self.getItems()
		tweets = {'dateTime': [], 'content': [],'y':[]}
		while(self.count):
			tweet = next(tweetGenerator)
			tweets['dateTime'].append(tweet.dateTime)
			tweets['content'].append(tweet.content)
			tweets['y'].append(0)
			self.count-=1
		return pandas.DataFrame(tweets)