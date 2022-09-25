import tweepy 
import time 
from kafka import KafkaProducer
import json
from textblob import TextBlob
from databaseOperations import db
from apscheduler.schedulers.blocking import BlockingScheduler
import schedule
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
# import spacy

api_key = 'xhL40tWv0tJr24Gih6lCUHQ8t'
api_secret = 'eBPPZGnvXrqhAMi9ZLnzQPcNkUtxjBNaJWXhFv9cYrOBG6uHwR'
bearer_token = r'AAAAAAAAAAAAAAAAAAAAANdDgAEAAAAA3X%2BTSEcuQ89ZO5YJdF1xURIRQ10%3DfSpTZgA205NzbS0MptKFKqFDnUioMs3ig7QftN3yVPcNO7yd4n'
access_token = '1559869967843184642-6rVhksiGSy7Pa3vDqQzh3xwSjZ9n5y'
access_token_secret = 'RxUIaalBlUoFvR3fPsoz6MTzgP4Prfj353XxgF8rM5FiE'

client = tweepy.Client(bearer_token, api_key, api_secret, access_token, access_token_secret)

auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)

api = tweepy.API(auth)

search_terms = ["Queen", "British"]

# def json_serializer(data):
#     return json.dumps(data).encode("utf-8")
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(lemma_words)

# def clean_up(text):
#     nlp = spacy.load("en_core_web_sm")
#     removal=['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE']
#     text_out = []
#     doc= nlp(text)
#     for token in doc:
#         if token.is_stop == False and token.is_alpha and token.pos_ not in removal:
#             lemma = token.lemma_
#             text_out.append(lemma)
#     return " ".join(text_out)

# producer = KafkaProducer(bootstrap_servers =['localhost:9092'], value_serializer=json_serializer, api_version=(2, 0, 2) )

class MyStream(tweepy.StreamingClient):
    #if time since start of stream is greater than 60 seconds, stop the stream
    
           
        
    def on_connect(self):
        print("Connected")
        self.num_tweets = 0
        
    def on_tweet(self, tweet):
        
        # if time since start of stream is less than 10 seconds
        
        if tweet.referenced_tweets == None:
            # print()
            # producer.send("SentimentAn",tweet.text) 
            print(tweet.text)
            # print(TextBlob(tweet.text).detect_language())
            
            Stopwords = stopwords.words('english')
            preprocessed = preprocess(tweet.text)
            
            
            analysis = TextBlob(preprocessed)
            senti= analysis.sentiment.polarity
            if senti<0 :
                emotion = 0
            elif senti>0:
                emotion= 1
            else:
                emotion = 0
            twitterfeed = {'results':preprocessed, 'sentiment':emotion}   
            
            self.num_tweets += 1
            
            if self.num_tweets < 10000:
                db.collection.insert_one(twitterfeed)
                time.sleep(0.2)
                return True
            else:
                self.disconnect()
                return False
            
            
            
    # create an on_status method to stop the stream after 10 seconds
    
    # def on_status(self):
    #     if self.num_tweets < 10:
    #         return True
    #     else:
    #         return False
            
stream = MyStream(bearer_token=bearer_token)

for term in search_terms:
    stream.add_rules(tweepy.StreamRule(term))
    
stream.filter(tweet_fields=["referenced_tweets"])




# automate running the script every 24 hours for 5 minutes

schedule.every(5).minutes.do(stream.filter, tweet_fields=["referenced_tweets"])

while True: 
    schedule.run_pending()
    time.sleep(1)

# scheduler = BlockingScheduler()
# scheduler.add_job(go, 'interval', minutes=5)
# scheduler.start()



       