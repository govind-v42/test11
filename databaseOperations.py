from pymongo import MongoClient
import pandas as pd
import json
import certifi

ca = certifi.where()

cluster = MongoClient("mongodb+srv://admin:admin@cluster0.ueloc2l.mongodb.net/test", tlsCAFile=ca)

db = cluster['Twittertestabc']

collection = db['collection']



# collection.create_index([('_id',1)], unique =True)

# collection.dropIndex("_id")


