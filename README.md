# **Dynamic Twitter Sentiment Analysis and Prediction**

## **About the Project**:

### The project has been undertaken to get streaming data (tweets) from twitter and create a deep learning model that would analyze the sentiment of the tweets. The work focuses solely on English language and the final output of the work is a web-app that would display the sentiment when the user inputs a tweet. The entire pipeline uses concepts such as data-version-control, containerization, and Kubernetes cluster deployment. The scripts can be broadly classified into two:- a streaming script (Streamer2.py) that can run round the clock to get the tweets, and the pipeline involving an ETL-job, a model-selection along with automatic retraining cript, and web-app part. All the scripts can be run locally. The Model accuracy tracking using Mlflow in the train-test.py requires authentication for Azure MLstudio. Set up new tracking uri if required. Also, Interactive Authentication with Azure MLStudio is not supported by the workflow design. Hence, MLStudio uri may be commented for the final execution of pipeline.  

## **Built With**: 
- tweepy
- TextBlob
- MongoDB
- petl
- sklearn
- mlflow
- flask
- dvc
- docker
- minikube
- pymongo

## **Pre-Requisites**
### Libraries to be installed:

- pip install dvc
- pip install sklearn
- pip install nltk
- pip install tweepy
- pip install pymongo[srv]
- pip install petl
- pip install mlflow
- pip install flask

## **Scripts**

1.	**databaseOperations.py** :  This module connects to the MongoDB cluster set up by the author.

2.	**Streamer2.py** : This module connects to twitter via api and retrieves tweets based on given search filter values. The tweets are pre-processed, annotated with a sentiment using TextBlob, and then sent to the MongoDB cluster using the databaseOperations.py module.

3.	**etlpipeline.py**: Extracts the data stored in MongoDB cluster and transforms the data using petl. In the transformation stage, the streaming tweets are combined with data from Sentiment140 dataset and a hybrid dataset for train-test is created. The data is then loaded into a csv format at root/data folder. 

4.	**Train-test.py** : This module takes in the hybrid dataset and splits the data for training and testing. A Naïve Bayes Bernoulli Classifier model is fitted to the data and the test stage prediction is evaluated. The accuracy score of the model is uploaded in Azure ML Studio using mlflow. The fitted model and vectorizer are dumped into pickle files. 

5.	**App.py**:- Creates a web-app based on python-flask which would take a tweet as an input and renders prediction and passed tweet as the final visual. This module takes in the pickle files from the previous stage

### Other Files

1.	**dvc.yaml** :  This file defines the first two-stages of the pipeline part of the project, viz, the ETL stage and the train-test stage. Dependencies for each stage and outputs for each stage have been mentioned. 

2.	**Dockerfile** : This file defines the two stages of the the pipeline part of the project required to build a Docker image.

3.	**Deployment.yaml** : This file contains the specifications, details of the number of pods, and Docker image location information for creating the Kubernetes cluster. 

4. **templates/index.html** : HTML file to render prediction of tweet in flask app

5. **data/data2.csv** : This is a truncated version of Static Twitter sentiment annotated dataset called Sentiment 140. Used in the pipeline to concatenate with dynamic incoming tweets and create a resulting hybrid dataset for model training and testing.  




