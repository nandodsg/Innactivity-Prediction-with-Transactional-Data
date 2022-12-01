# Innactivity Prediction with Transacional Data

### Table of Content
1. [Overview](#overview)
2. [Instalation](#instalation)
3. [File descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors and Aknowledments](#licensing)

## 1. Overview <a name="overview"></a>
This project aimed at analyzing the innactivy prediction measured by the number of transaction credit union members perform on the institutions mobile app. The idea behind the project is to provide further insight into churn predictors based solely on transacional records.

This project uses Logistic Regression and Random Tree classifiers to test the prediction. Seeing as the dataset is highly inbalanced (activity is way higher than innactivity) a few resampling techniques were tested to try the find the best model fit.

The whole project followed the CRISP-DM Data Science method. Check out this post for a detailed explanation.

## 2. Instalation <a name="instalation"></a>

This project uses opensource Python libraries: Numpy, Pandas, Matplotlib, SKLearn and Imblearn

## 3. File descriptions <a name="files"></a>

All datasets were anomyzed by the author using pseudo IDs.
pseudo SQL querys are presented with dataset specificity omitted due to confidentiallity.

Files used during the exploratory analysis phase:

exploratory analysis.pbix -- Power BI files
pseudo query - exploratory analysis dataset (anonymous).sql -- SQL Algorithm for creating the accompanying dataset.
exploratory analysis dataset (anonymous).csv -- Analytical transacional dataset for 981 credit union members
exporatory analysis - churn flags (anonymous).csv -- Innactivity prediction dataset for 981 credit union members
pseudo query - exporatory analysis - churn flags (anonymous).sql -- SQL Algorithm for creating the accompanying dataset.
 
Files used model development and testing:

pseudo query.sql-- SQL Algorithm for creating the accompanying dataset.
Model Data Set (pseudo).csv -- Dataset used for the model
Innactivity prediction with transactional data.ipynb -- Model development and evaluation python notebook

## 4. Results <a name="results"></a>

Though the exploratory analysis indicates the possibily of finding correlation between transaction patterns and innactivity, the two classifiers and 4 resampling techniques used did not present good performance on this highly imbalanced dataset.

### Recommendations on future studies

1. Study the use of time series prediction techniques as a subsititue for Classifiers
2. Use the accumlative transactional variation on 5 months prior to the 6th month innactivity prediction may wielf better results than using the absolute number of transations per month as features.

## 5. Licensing, Authors and Aknowledments <a name="licensing"></a>

I've realesed this project as Creative Commons.
The original data comes from Sicredi Credit Union, from Brazil.
