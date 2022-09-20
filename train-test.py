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
from sklearn.linear_model import LogisticRegression

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

region = "Germany West Central"
subscription_id = "ad24f89b-ce25-48f7-89af-5f742bad090d"
resource_group = "Govind"
workspace_name = "Twitter"

azureml_mlflow_uri = f"azureml://germanywestcentral.api.azureml.ms/mlflow/v1.0/subscriptions/ad24f89b-ce25-48f7-89af-5f742bad090d/resourceGroups/Govind/providers/Microsoft.MachineLearningServices/workspaces/Twitter"
mlflow.set_tracking_uri(azureml_mlflow_uri)
experiment_name = 'mlflowdemo'
mlflow.set_experiment(experiment_name)

if __name__ == "__main__":
    # mlflow.set_experiment(experiment_name="mlflowdemo")
    # region = "Germany West Central"
    # subscription_id = "ad24f89b-ce25-48f7-89af-5f742bad090d"
    # resource_group = "Govind"
    # workspace_name = "Twitter"
    
    # azureml_mlflow_uri = f"azureml://germanywestcentral.api.azureml.ms/mlflow/v1.0/subscriptions/ad24f89b-ce25-48f7-89af-5f742bad090d/resourceGroups/Govind/providers/Microsoft.MachineLearningServices/workspaces/Twitter"
    # mlflow.set_tracking_uri(azureml_mlflow_uri)
    
    
    # Const
    DATASET_COLUMNS = [ "_id", "results", "sentiment"]
    DATASET_ENCODING = "ISO-8859-1"
    TARGET_COL = 'sentiment'
    CSV_PATH = 'data/data3.csv'
    NEW_DIR = 'split-data'
    # X_TRAIN_PATH = 'split-data/X_train.csv'
    # X_TEST_PATH = 'split-data/y_train.csv'
    # Y_TRAIN_PATH = 'split-data/X_test.csv'
    # Y_TEST_PATH = 'split-data/y_test.csv'
    TOT_SIZE = 200000
    TEST_SIZE = 0.2

    print("Read raw data")
    df = pd.read_csv(CSV_PATH, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
    # print(f'data set shape {df.shape}')


    data=df[['results','sentiment']]

    print(df.head())
    # lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer()
    # def preprocess(sentence):
    #     sentence=str(sentence)
    #     sentence = sentence.lower()
    #     sentence=sentence.replace('{html}',"") 
    #     cleanr = re.compile('<.*?>')
    #     cleantext = re.sub(cleanr, '', sentence)
    #     rem_url=re.sub(r'http\S+', '',cleantext)
    #     rem_num = re.sub('[0-9]+', '', rem_url)
    #     tokenizer = RegexpTokenizer(r'\w+')
    #     tokens = tokenizer.tokenize(rem_num)  
    #     filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    #     stem_words=[stemmer.stem(w) for w in filtered_words]
    #     lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    #     return " ".join(lemma_words)

    # data['results'] = data['results'].apply(lambda text: preprocess(text))

    # data['target']= data['target'].astype(int)

    # print(data['target'] == 4)

    # data['target'] = data['target'].replace(4,1)

    # data_pos = data[data['target'] == 1]



    # # data_neg = data[data['target'] == 0]

    # # data_pos = data_pos.iloc[:int(2500)]
    # # data_neg = data_neg.iloc[:int(2500)]

    # # dataset = pd.concat([data_pos, data_neg])

    # # dataset['text']=dataset['text'].str.lower()

    # # stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
    # #              'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
    # #              'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
    # #              'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
    # #              'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
    # #              'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
    # #              'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
    # #              'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
    # #              'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
    # #              't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
    # #              'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
    # #              'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
    # #              'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
    # #              'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
    # #              "youve", 'your', 'yours', 'yourself', 'yourselves']

    # # STOPWORDS = set(stopwordlist)
    # # def cleaning_stopwords(text):
    # #     return " ".join([word for word in str(text).split() if word not in STOPWORDS])
    # # dataset['text'] = dataset['text'].apply(lambda text: cleaning_stopwords(text))


    # # english_punctuations = string.punctuation
    # # punctuations_list = english_punctuations
    # # def cleaning_punctuations(text):
    # #     translator = str.maketrans('', '', punctuations_list)
    # #     return text.translate(translator)
    # # dataset['text']= dataset['text'].apply(lambda x: cleaning_punctuations(x))


    # # def cleaning_repeating_char(text):
    # #     return re.sub(r'(.)1+', r'1', text)
    # # dataset['text'] = dataset['text'].apply(lambda x: cleaning_repeating_char(x))

    # # def cleaning_URLs(data):
    # #     return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
    # # dataset['text'] = dataset['text'].apply(lambda x: cleaning_URLs(x))

    # # def cleaning_numbers(data):
    # #     return re.sub('[0-9]+', '', data)
    # # dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))


    # # tokenizer = RegexpTokenizer(r'w+')
    # # dataset['text'] = dataset['text'].apply(tokenizer.tokenize)


    # # st = nltk.PorterStemmer()
    # # def stemming_on_text(data):
    # #     text = [st.stem(word) for word in data]
    # #     return data
    # # dataset['text']= dataset['text'].apply(lambda x: stemming_on_text(x))

    # # lm = nltk.WordNetLemmatizer()
    # # def lemmatizer_on_text(data):
    # #     text = [lm.lemmatize(word) for word in data]
    # #     return data
    # # dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x))



    X=data.results
    y=data.sentiment

    # data_neg = data['text'][:800000]

    # # Separating the 95% data for training data and 5% for testing data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state =42)



    # # print("Replace target col values")
    # # df[TARGET_COL] = df[TARGET_COL].replace(0, 1)  # Negative
    # # df[TARGET_COL] = df[TARGET_COL].replace(4, 0)  # Positive

    # # os.makedirs(NEW_DIR, exist_ok=True)

    # # print("Split dataset to train and test")
    # # X_train, X_test, y_train, y_test = train_test_split(df.drop(TARGET_COL, axis=1), df[TARGET_COL],
    # #                                                     test_size=TEST_SIZE, random_state=42,
    # #                                                     stratify=df[TARGET_COL])

    # # print("Save data sets to csv")
    # # X_train.to_csv(X_TRAIN_PATH, index=False)
    # # y_train.to_csv(X_TEST_PATH, index=False)
    # # X_test.to_csv(Y_TRAIN_PATH, index=False)

    # # # y_test will be saved outside of the repo - to prevent cheating.
    # # y_test.to_csv(Y_TEST_PATH, index=False)




    vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000, lowercase=False)

    vectoriser.fit(X_train.values.astype('U'))


    print('No. of feature_words: ', len(vectoriser.get_feature_names()))

    X_train = vectoriser.transform(X_train.values.astype('U'))
    X_test  = vectoriser.transform(X_test.values.astype('U'))

    pickle.dump(vectoriser,open("feature1.pkl","wb"))

    def model_Evaluate(model):
        # Predict values for Test dataset
        y_pred = model.predict(X_test)
        # Print the evaluation metrics for the dataset.
        print(classification_report(y_test, y_pred))
        # Compute and plot the Confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
        categories = ['Neutral','Positive']
        group_names = ['True Neg','False Pos', 'False Neg','True Pos']
        group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
        labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
        labels = np.asarray(labels).reshape(-2,2)
        sns.heatmap(cf_matrix, cmap = 'Blues',fmt = '', xticklabels = categories, yticklabels = categories)
        plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
        plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
        plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
        plt.show()

    
    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train, y_train)
    y_pred1 = BNBmodel.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred1)
    
    with mlflow.start_run() as mlflow_run:
        mlflow.log_metric("Accuracy", model_accuracy)
        mlflow.sklearn.log_model(BNBmodel, "model")
        
    
    # tweet = ['Queen was very very happy and a very sweet sunshine']
    # lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer() 
    # def preprocess(sentence):
    #     sentence=str(sentence)
    #     sentence = sentence.lower()
    #     sentence=sentence.replace('{html}',"") 
    #     cleanr = re.compile('<.*?>')
    #     cleantext = re.sub(cleanr, '', sentence)
    #     rem_url=re.sub(r'http\S+', '',cleantext)
    #     rem_num = re.sub('[0-9]+', '', rem_url)
    #     tokenizer = RegexpTokenizer(r'\w+')
    #     tokens = tokenizer.tokenize(rem_num)  
    #     filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    #     stem_words=[stemmer.stem(w) for w in filtered_words]
    #     lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    #     return " ".join(lemma_words)

    # y_pred2 = BNBmodel.predict(vectoriser.transform(tweet))
    # print(y_pred2)
    # model_Evaluate(BNBmodel)


    filename = 'finalized_model1.pkl'
    pickle.dump(BNBmodel, open(filename, 'wb'))

