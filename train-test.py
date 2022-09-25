from ctypes.wintypes import SIZE
from dataclasses import replace
import pandas as pd
from sklearn.model_selection import train_test_split
import os
# utilities
import re
import numpy as np
import pandas as pd
# plotting
import seaborn as sns

import matplotlib.pyplot as plt
# nltk
from nltk.stem import WordNetLemmatizer
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import string
from nltk.tokenize import RegexpTokenizer
import nltk
import pickle
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,KFold







# mlflow.set_experiment(experiment_name="mlflowdemo")
# region = "Germany West Central"
# subscription_id = "ad24f89b-ce25-48f7-89af-5f742bad090d"
# resource_group = "Govind"
# workspace_name = "Twitter"

# azureml_mlflow_uri = f"azureml://germanywestcentral.api.azureml.ms/mlflow/v1.0/subscriptions/ad24f89b-ce25-48f7-89af-5f742bad090d/resourceGroups/Govind/providers/Microsoft.MachineLearningServices/workspaces/Twitter"
# mlflow.set_tracking_uri(azureml_mlflow_uri)
# region = "Germany West Central"
# subscription_id = "ad24f89b-ce25-48f7-89af-5f742bad090d"
# resource_group = "Govind"
# workspace_name = "Twitter"
# os.environ["MLFLOW_TRACKING_USERNAME"]= "11016106@stud.hochschule-heidelberg.de"

# os.environ["MLFLOW_TRACKING_PASSWORD"] = "Medical1234"

# os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

# os.environ['MLFLOW_TRACKING_TOKEN'] = 

# azureml_mlflow_uri = f"azureml://germanywestcentral.api.azureml.ms/mlflow/v1.0/subscriptions/ad24f89b-ce25-48f7-89af-5f742bad090d/resourceGroups/Govind/providers/Microsoft.MachineLearningServices/workspaces/Twitter"

# mlflow.set_tracking_uri(azureml_mlflow_uri)

# experiment_name = 'mlflowdemo'
# mlflow.set_experiment(experiment_name)

# if __name__ == "__main__":


DATASET_COLUMNS = [ "results", "sentiment"]
DATASET_ENCODING = "ISO-8859-1"
TARGET_COL = 'sentiment'
CSV_PATH = 'data/data5.csv'
NEW_DIR = 'split-data'
# X_TRAIN_PATH = 'split-data/X_train.csv'
# X_TEST_PATH = 'split-data/y_train.csv'
# Y_TRAIN_PATH = 'split-data/X_test.csv'
# Y_TEST_PATH = 'split-data/y_test.csv'

# TEST_SIZE = 0.2

print("Read raw data")
df = pd.read_csv(CSV_PATH, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1,)
# print(f'data set shape {df.shape}')


data=df[['results','sentiment']]

data_pos = data[data['sentiment'] == 1]

data_neg = data[data['sentiment'] == 0]
data_pos = data_pos.sample(3000)
data_neg = data_neg.sample(3000)
dataset = pd.concat([data_pos, data_neg])

X2 = dataset['results']
y2 = dataset['sentiment']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2,test_size = 0.20, random_state =42)
vectoriser2 = TfidfVectorizer(ngram_range=(1,2), max_features=500000, lowercase=False)

vectoriser2.fit(X_train2.values.astype('U'))
vectoriser2.fit(X_train2.values.astype('U'))
print('No. of feature_words2: ', len(vectoriser2.get_feature_names_out()))

X_train2= vectoriser2.transform(X_train2.values.astype('U'))
X_test2  = vectoriser2.transform(X_test2.values.astype('U'))








kf=KFold(n_splits=5)
log_reg_params = [{"C":0.01}, {"C":0.1}, {"C":1}, {"C":10}]
dec_tree_params = [ {"criterion": "entropy"}]
rand_for_params = [{"criterion": "gini"}, {"criterion": "entropy"}]
kneighbors_params = [{"n_neighbors":3}, {"n_neighbors":5}]
naive_bayes_params = [{}]
svc_params = [{"C":0.01}, {"C":0.1}, {"C":1}, {"C":10}]

modelclasses = modelclasses = [
    ["log regression", LogisticRegression, log_reg_params],
    ["decision tree", DecisionTreeClassifier, dec_tree_params],
    ["random forest", RandomForestClassifier, rand_for_params],
    ["k neighbors", KNeighborsClassifier, kneighbors_params],
    ["naive bayes", BernoulliNB, naive_bayes_params],
    ["support vector machines", SVC, svc_params]
]
insights = []
for modelname, Model, params_list in modelclasses:
    for params in params_list:
        model = Model(**params)
        score=cross_val_score(model,X_train2,y_train2,cv=kf)
        print("Cross Validation Scores are {}".format(score))
        print("Average Cross Validation score :{}".format(score.mean()))
        truescore = score.mean()
        insights.append((modelname, model, params, truescore))


insights.sort(key=lambda x:x[-1], reverse=True)

print(insights)
print (insights[0][1])




X=data.results
y=data.sentiment


# # # Separating the 95% data for training data and 5% for testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state =42)

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000, lowercase=False)

vectoriser.fit(X_train.values.astype('U'))


print('No. of feature_words: ', len(vectoriser.get_feature_names_out()))

X_train = vectoriser.transform(X_train.values.astype('U'))
X_test  = vectoriser.transform(X_test.values.astype('U'))

pickle.dump(vectoriser,open("feature1.pkl","wb"))

model = insights[0][1]

model.fit( X_train, y_train)
y_pred1 = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred1)

print(model_accuracy)

    
filename = 'finalized_model1.pkl'
pickle.dump(model, open(filename, 'wb'))





# with mlflow.start_run() as mlflow_run:
#     mlflow.log_metric("Accuracy", model_accuracy)
#     mlflow.sklearn.log_model(model, "model")
    
    






    

