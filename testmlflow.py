import mlflow
import os
from random import random

region = "Germany West Central"
subscription_id = "ad24f89b-ce25-48f7-89af-5f742bad090d"
resource_group = "Govind"
workspace_name = "Twitter"

azureml_mlflow_uri = f"azureml://germanywestcentral.api.azureml.ms/mlflow/v1.0/subscriptions/ad24f89b-ce25-48f7-89af-5f742bad090d/resourceGroups/Govind/providers/Microsoft.MachineLearningServices/workspaces/Twitter"
mlflow.set_tracking_uri(azureml_mlflow_uri)

experiment_name = 'experiment_with_mlflow'
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as mlflow_run:
    mlflow.log_param("hello_param", "world")
    mlflow.log_metric("hello_metric", random())
    os.system(f"echo 'hello world' > helloworld.txt")
    mlflow.log_artifact("helloworld.txt")