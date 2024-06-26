# Loan Data Analysis and Return Rate Prediction

## Introduction
This project focuses on analyzing and predicting loan returns using various machine learning algorithms and mlflow. The workflow involves data preprocessing, exploratory data analysis (EDA), and the testing of models with mlflow. For more information and the results you can browse: https://www.canva.com/design/DAGFrboTucs/qc7oJvzpvOsHuzKhrcugzg/edit?utm_content=DAGFrboTucs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton 

## Experiment Structure
I wanted to simulate a real data flow, so I divided the dataset into four batches that have similar distributions among the variables. Then, I used SMOTE to balance the distribution of the dependent variable, not.fully.paid, which indicates whether the loan has been fully paid back or not.  For every 4 batches, I ran 6 experiments and I also did hyperparameters search by using hyperopt, along with testing for the default hyperparameters. 

Each algorithm ran twice (with and without hyperopt search space) for 4 batches, making a total of 24 runs. The algorithms I used are; 


* Decision Tree  

* Random Forest

* XGBoost 




```zsh
mlflow server --port 8080
```

then run file main.py

Dataset is taken from : https://www.kaggle.com/datasets/saramah/loan-data

