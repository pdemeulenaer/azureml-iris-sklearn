#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Config file already exist in compute instance 
from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')


# In[2]:


from azureml.core import Experiment
experiment_name = 'example-iris-sklearn-experiment'
experiment = Experiment(workspace = ws, name = experiment_name)
experiment


# In[16]:


import os
import sys
import json
import socket
import traceback
import pandas as pd
import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import pylab
from pylab import *
import matplotlib.cm as cm
import matplotlib.mlab as mlab

#Import of SKLEARN packages
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

import mlflow
from mlflow.models.signature import infer_signature
from azureml.core import Workspace, Datastore, Dataset


# In[4]:


data_json = '''{
    "TEST": {
        "input_train": "default.iris",
        "input_test": "default.iris_test",
        "output_test": "default.iris_test_scored",
        "input_to_score": "default.iris_to_score",
        "output_to_score": "default.scored"  
    },
    "SYST": {
        "input_train": "test.iris",
        "input_test": "test.iris_test",
        "output_test": "test.iris_test_scored",
        "input_to_score": "test.iris_to_score",
        "output_to_score": "test.scored"        
    },
    "PROD": {
        "input_train": "test.iris",
        "input_test": "test.iris_test",
        "output_test": "test.iris_test_scored",
        "input_to_score": "test.iris_to_score",
        "output_to_score": "test.scored"       
    }
}'''

config_json = '''{
    "hyperparameters": {
        "max_depth": "20",
        "n_estimators": "100",
        "max_features": "auto",
        "criterion": "gini",
        "class_weight": "balanced",
        "bootstrap": "True",
        "random_state": "21"        
    }
}'''

data_conf = json.loads(data_json)
model_conf = json.loads(config_json)

data_conf, model_conf


# In[14]:


# ==============================
# DATA MAKING + REGISTERING IN DATA STORE (blob one)
# ==============================

# Here we will create Datasets from Pandas dataframe (https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets#create-a-dataset-from-pandas-dataframe)

# Loading of dataset
iris = load_iris()                  #The Iris dataset is available through the scikit-learn API
idx = list(range(len(iris.target)))
np.random.shuffle(idx)              #We shuffle it (important if we want to split in train and test sets)
X = iris.data[idx]
y = iris.target[idx]
# Load data in Pandas dataFrame
data_pd = pd.DataFrame(data=np.column_stack((X,y)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
data_pd.loc[data_pd['label']==0,'species'] = 'setosa'
data_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
data_pd.loc[data_pd['label']==2,'species'] = 'virginica'

datastore = Datastore.get(ws, 'workspaceblobstore')
iris_dataset = Dataset.Tabular.register_pandas_dataframe(data_pd, datastore, "iris_all", show_progress=True)
# iris_dataset = iris_dataset.register(workspace=ws, name='iris_dataset', description='iris all data')

# Feature selection
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target       = 'label'   

X = data_pd[feature_cols].values
y = data_pd[target].values
# Creation of train and test datasets
x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.7, stratify=y) #stratify=y ensures that the same proportion of labels are in both train and test sets! 

# Save test dataset
test_pd = pd.DataFrame(data=np.column_stack((x_test,y_test)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
test_pd.loc[data_pd['label']==0,'species'] = 'setosa'
test_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
test_pd.loc[data_pd['label']==2,'species'] = 'virginica'
# test_df = spark.createDataFrame(test_pd)
# test_df.write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format('test_data_sklearn_rf'))

datastore = Datastore.get(ws, 'workspaceblobstore')
test_dataset = Dataset.Tabular.register_pandas_dataframe(test_pd, datastore, "test_pd", show_progress=True)
# test_dataset = test_dataset.register(workspace=ws, name='test_dataset', description='iris test data')


print("Step 1.0 completed: Loaded Iris dataset in Pandas")  


# In[12]:


# Available data stores:
datastores = ws.datastores
datastores


# In[17]:


# DEFINE AND TRAIN MODEL

# create MLflow experiment and start logging to a new run in the experiment
experiment_name = "simple-rf-sklearn_experiment"
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)
# mlflow.autolog()

with mlflow.start_run() as run:  

    # Model definition
    max_depth = int(model_conf['hyperparameters']['max_depth'])
    n_estimators = int(model_conf['hyperparameters']['n_estimators'])
    max_features = model_conf['hyperparameters']['max_features']
    criterion = model_conf['hyperparameters']['criterion']
    class_weight = model_conf['hyperparameters']['class_weight']
    bootstrap = bool(model_conf['hyperparameters']['bootstrap'])
    clf = RandomForestClassifier(max_depth=max_depth,
                            n_estimators=n_estimators,
                            max_features=max_features,
                            criterion=criterion,
                            class_weight=class_weight,
                            bootstrap=bootstrap,
                            random_state=21,
                            n_jobs=-1)          

    # Fit of the model on the training set
    model = clf.fit(x_train, y_train) 

    # Log the model within the MLflow run
    mlflow.log_param("max_depth", str(max_depth))
    mlflow.log_param("n_estimators", str(n_estimators))  
    mlflow.log_param("max_features", str(max_features))             
    mlflow.log_param("criterion", str(criterion))  
    mlflow.log_param("class_weight", str(class_weight))  
    mlflow.log_param("bootstrap", str(bootstrap))  
    mlflow.log_param("max_features", str(max_features)) 
    signature = infer_signature(x_train, clf.predict(x_train))
    
    input_example = {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
    
    mlflow.sklearn.log_model(model, 
                            "model",
                            registered_model_name="sklearn-rf",
                            signature=signature,
                            input_example=input_example)   


# # Deploy the model for real-time inference

# In[18]:


# Create deployment config for ACI

# create environment for the deploy
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import AciWebservice

# get a CURATED environment  !! WE NEED TO USE AN IMAGE
env = Environment.get(
    workspace=ws, 
    name="AzureML-sklearn-0.24.1-ubuntu18.04-py37-cpu-inference",
    version=1
)
env.inferencing_stack_version='latest'

# create deployment config i.e. compute resources
aciconfig = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    tags={"data": "IRIS", "method": "sklearn"},  #????
    description="Predict IRIS with sklearn",
)


# # Deploy model
# This next code cell deploys the model to Azure Container Instance (ACI).
# 
# Note: The deployment takes approximately 3 minutes to complete.

# In[24]:


get_ipython().run_cell_magic('time', '', 'import uuid\nfrom azureml.core.model import InferenceConfig\nfrom azureml.core.environment import Environment\nfrom azureml.core.model import Model\n\n# get the registered model\nmodel = Model(ws, "sklearn-rf")\n\n# create an inference config i.e. the scoring script and environment\ninference_config = InferenceConfig(entry_script="score.py", environment=env)\n\n# deploy the service\nservice_name = "sklearn-iris-svc-" + str(uuid.uuid4())[:4]\nservice = Model.deploy(\n    workspace=ws,\n    name=service_name,\n    models=[model],\n    inference_config=inference_config,\n    deployment_config=aciconfig,\n)\n\nservice.wait_for_deployment(show_output=True)')


# In[26]:


# send raw HTTP request to test the web service.
import requests

# send a random row from the test set to score
random_index = np.random.randint(0, len(x_test) - 1)
input_data = '{"data": [' + str(list(x_test[random_index])) + "]}"

headers = {"Content-Type": "application/json"}

resp = requests.post(service.scoring_uri, input_data, headers=headers)

print("POST to url", service.scoring_uri)
print("label:", y_test[random_index])
print("prediction:", resp.text)


# In[21]:


random_index = np.random.randint(0, len(x_test) - 1)
input_data = '{"data": [' + str(list(x_test[random_index])) + "]}"
input_data


# In[ ]:




