# from flair.models import TextClassifier
# from flair.data import Sentence
from dataclasses import replace
import pandas as pd
import petl as etl
from pymongo import MongoClient
import certifi
from langdetect import detect, DetectorFactory

ca = certifi.where()

cluster = MongoClient("mongodb+srv://admin:admin@cluster0.ueloc2l.mongodb.net/test", tlsCAFile=ca)

db = cluster['NewDB']

collection = db['collection']

df = pd.DataFrame(list(collection.find()))


DetectorFactory.seed = 0

table = etl.fromdataframe(df)
table1 = etl.cutout(table, '_id')



tablea = etl.fromcsv('data/data2.csv', encoding="utf8")
tableb = etl.cutout(tablea, '2228970737','NO_QUERY', 'Thu Jun 18 15:07:03 PDT 2009','bitterdreams')
tablec = etl.rename(tableb, { 'http://twitpic.com/7qvcb - I WANT CANDYS â™¥ ': 'results',  '\ufeff0': 'sentiment'})
tabled = etl.transform.headers.sortheader(tablec)

# tablec = etl.cut(tableb, 'http://twitpic.com/7qvcb - I WANT CANDYS Ã¢â„¢Â¥ ', 'ï»¿0')
# tabled = etl.convert(tablec, 'ï»¿0', 'replace', 4, 1)
# df1 = etl.todataframe(tableb)
# print(df1.head())
print(etl.fieldnames(tabled))
tablee = etl.convert(tabled, 'sentiment', 'replace', '4', '1')

# tabled = etl.convert(tabled, 'sentiment', 'replace', 4, 1)

print(tablee)
# table2 = etl.cutout(table, '_id', 'sentiment')

# print(table2)

table3 = etl.cat(table1, tablee)

df = etl.todataframe(table3)

df['sentiment'] = df['sentiment'].astype(int)

df.to_csv('data/data5.csv', index=False)

# def get_language(tweet):
#     try:                                                          
#         lang=detect(tweet)                                      
#     except:                                                       
#         lang='no'
         
#     return lang

# df = pd.read_csv('data/data3.csv') 

# print(table2['results'])

# print(get_language)

# table4 = etl.addfields(table3,('lang', get_language),('results'))
# print (table4)

# df['lang'] = df['results'].apply(lambda x: get_language(x))

# df = df[df['lang'] == 'en']

# df.to_csv('data/data5.csv', index=False)

# print(table3)

# etl.tocsv(table2, 'example.csv')


# df1 = df[df.lang == 'en']

# df1.to_csv('data/data4.csv', index=False)

# sia = TextClassifier.load('en-sentiment')
# def flair_prediction(x):
#     # sentence = Sentence(x)
#     sia.predict(x)
#     score = x.labels[0]
#     if "POSITIVE" in str(score):
#         return 1
#     elif "NEGATIVE" in str(score):
#         return -1
#     else:
#         return 0
    
# df = pd.read_csv('data/data3.csv')    




# df["sentimentflair"] = df["results"].apply(flair_prediction)

# df.to_csv('data/data4.csv', index=False)
