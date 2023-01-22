# Introduction

Objective of the project is to predict the salary status of a user as high (>50K) or low (<=50K) based on experience, age, education, marital status and demographics. 

Data used for the model is publicly available census data

# Installation

Project uses the python version 3.10.9. Before executing the code it is require to have the required packages installed. Project repository includes requirements.txt file which specifies the required packages. Below are the steps to install the required packages.

<code> pip install -r requirements.txt </code>


# Execution

<h5>To train the model, scripts are in src directory</h5>

<code> python src/train_model.py</code>


# API

<div >
API is developed using FastAPI and is depoloyed at heroku platform. Link for the app is below
<a>https://salary-prediction.herokuapp.com/</a>
</div>

# Continous Integration

Project uses the github actions to perform the continous integration. Steps include the testing using pytest. Tests considered before build are:
   - test of API
   - test of model performance overall
   - test of model performance on slices of categorical data


# Continous Deployment

For CD model uses the Heroku continous deployment which triggers itself after github build.

