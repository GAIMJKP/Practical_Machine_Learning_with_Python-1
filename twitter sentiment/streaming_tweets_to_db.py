from tweepy import Stream
import tweepy
from tweepy.streaming import StreamListener
import sqlite3
import time
import json
from unidecode import unidecode
from textblob import TextBlob



#replace mysql.server with "localhost" if you are running via your own server!
#                        server       MySQL username	MySQL pass  Database name.
# conn = MySQLdb.connect("mysql.server","beginneraccount","cookies","beginneraccount$tutorial")
# c = conn.cursor()
conn = sqlite3.connect('twitter.db')
c = conn.cursor()




#consumer key, consumer secret, access token, access secret.
consumer_key= ''
consumer_secret= ''
access_token=''
access_token_secret= ''

api=None

conn = sqlite3.connect('twitter.db')
c = conn.cursor()

def create_table():
    try:
        c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, tweet TEXT, sentiment REAL)")
        c.execute("CREATE INDEX fast_unix ON sentiment(unix)")
        c.execute("CREATE INDEX fast_tweet ON sentiment(tweet)")
        c.execute("CREATE INDEX fast_sentiment ON sentiment(sentiment)")
        conn.commit()
    except Exception as e:
        print(str(e))
#create_table()

def connect():
	try:
		auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		auth.set_access_token(access_token, access_token_secret)
		global api
		api = tweepy.API(auth)
	except:
		print("Error: Authetication Failed")

#connect()

class listener(StreamListener):

	def on_data(self, data):
		try:
			data = json.loads(data)
			#print(data.keys())
			tweet = unidecode(data['text'])
			time_ms = data['timestamp_ms']
			analysis = TextBlob(tweet)
			vs = analysis.sentiment
			print(vs)
			sentiment = vs.polarity
			print(time_ms, tweet, sentiment)
			c.execute("INSERT INTO sentiment (unix, tweet, sentiment) VALUES (?, ?, ?)",
				(time_ms, tweet, sentiment))
			conn.commit()

		except KeyError as e:
			print(str(e))
		return(True)

	def on_error(self, status):
		print(status)


while True:
	try:
		# No puedo utilizar connect() por los hilos
		auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		auth.set_access_token(access_token, access_token_secret)
		twitterStream = Stream(auth, listener())
		twitterStream.filter(track=["a","e","i","o","u"])

	except Exception as e:
		print(str(e))
		time.sleep(5)