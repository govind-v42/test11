# from flair.models import TextClassifier
# from flair.data import Sentence
from dataclasses import replace
import pandas as pd
import petl as etl
from pymongo import MongoClient
import certifi


ca = certifi.where()

cluster = MongoClient("mongodb+srv://admin:admin@cluster0.ueloc2l.mongodb.net/test", tlsCAFile=ca)

db = cluster['NewDB']

collection = db['collection']

df = pd.DataFrame(list(collection.find()))

# data_pos = data[data['sentiment'] == 1]

# data_neg = data[data['sentiment'] == 0]
# data_pos = data_pos.sample(3000)
# data_neg = data_neg.sample(3000)

# DetectorFactory.seed = 0

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
tabled = etl.select(tablee, 'sentiment', lambda v: v == '1')
tablef = etl.head(tabled, 7500)
tableg = etl.select(tablee, 'sentiment', lambda v: v == '0')
tableh = etl.head(tableg, 7500)

tablei = etl.cat(tablef, tableh)


# tabled = etl.convert(tabled, 'sentiment', 'replace', 4, 1)

print(tablei)
# table2 = etl.cutout(table, '_id', 'sentiment')

# print(table2)

table3 = etl.cat(table1, tablei)

df = etl.todataframe(table3)

df['sentiment'] = df['sentiment'].astype(int)

df.to_csv('data/data5.csv', index=False)
