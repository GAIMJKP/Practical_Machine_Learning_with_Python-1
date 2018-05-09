# pip install textblob vadersentiment
from textblob import TextBlob
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy
import re
from unidecode import unidecode


#Keys and tokens from Twitter
consumer_key= ''
consumer_secret= ''
access_token=''
access_token_secret= ''
api=None


#Hashtag related to the debate
search = "Barcelona" 
#Date of the debate : October 13th and max tweet retrieved
since_date = "2017-08-20"
until_date = "2017-08-21"
num_tweet_max = 5
user_name ="BarackObama"


# Step 1 - Authenticate
def connect():
	try:
		auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		auth.set_access_token(access_token, access_token_secret)
		global api
		api = tweepy.API(auth)
	except:
		print("Error: Authetication Failed")

def get_tweet_sentiment(tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

#Cleaning the tweet
def clean_tweet(tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_public_tweets():
	#Step 3 - Retrieve Tweets
	public_tweets = api.search(search,count=num_tweet_max, since = since_date, until=until_date)
	for tweet in public_tweets:
	    print(tweet.text)	    
	    #Step 4 Perform Sentiment Analysis on Tweets
	    analysis = TextBlob(tweet.text)
	    print(analysis.sentiment)
	    print(get_tweet_sentiment(tweet.text))
	    print("")

def get_user_tweets():
	#Step 3 - Retrieve Tweets
	#user_tweets = api.user_timeline(user_name,count=num_tweet_max, tweet_mode='extended')
	user_tweets = api.home_timeline(count=num_tweet_max, tweet_mode='extended')
	for tweet in user_tweets:

		print(unidecode(tweet.full_text))
		print(clean_tweet(tweet.full_text))

	    #Step 4 Perform Sentiment Analysis on Tweets
	    analysis = TextBlob(unidecode(tweet.text))
	    print(analysis.sentiment)
	    print(get_tweet_sentiment(clean_teet(tweet.text)))
		print("")


if __name__ == '__main__':
	connect()
	get_user_tweets()

"""
analysis = TextBlob("TextBlob sure look like it has some interesting features!")
#print(dir(analysis))
print(analysis.translate(to='it')) # Esta detectando el fichero como español y no deja traducir a "es"
print(analysis.tags)
print(analysis.sentiment)  # Polarity -1 neg 1 pos  subjectivity 0-1  0 objective 1 subjective

pos_count = 0
pos_correct = 0

with open("positive.txt","r") as f:
    for line in f.read().split('\n'):
        analysis = TextBlob(line)
        if analysis.sentiment.polarity > 0:
            pos_correct += 1
        pos_count +=1


neg_count = 0
neg_correct = 0

with open("negative.txt","r") as f:
    for line in f.read().split('\n'):
        analysis = TextBlob(line)
        if analysis.sentiment.polarity <= 0:
            neg_correct += 1
        neg_count +=1

print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))
"""