from pymongo import MongoClient
import pandas as pd
import json
import certifi
import schedule

ca = certifi.where()

cluster = MongoClient("mongodb+srv://admin:admin@cluster0.ueloc2l.mongodb.net/test", tlsCAFile=ca)

db = cluster['NewDB']

collection = db['collection']

df = pd.DataFrame(list(collection.find()))

df.to_csv('data/data3.csv', index=False)




# collection.create_index([('_id',1)], unique =True)

# collection.dropIndex("_id")


