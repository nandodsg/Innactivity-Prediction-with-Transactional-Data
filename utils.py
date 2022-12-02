# Basic data wranging and plotting
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
# Creating the modeling dataset
from sklearn.datasets import make_classification
# Data processing
import pandas as pd
import numpy as np
# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Model and performance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
# Oversampling and under sampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter
# Processing time
import time

def model_predict(model_name,resampling_name,X,y,random_state,test_size,verbose):
    """
    Trains and Fits Models with different Resampling Techniques
    
    INPUT
    model_name = Str with model name
    resampling_name = Str with resampling technique name
    X = DataFrame with independent variables
    y = Vector with dependent (response) variable
    random_state = INT random state number
    verbose = STR to switch execution log on or off
    
    OUTPUT
    model_prediction = Vector with chosen model with resampling predictions on the test set
    """
    start_time = time.time() #Count processing time
    print('\n--------------------------------------------------------------------------------\n--------------------------------------------------------------------------------')
    if verbose != 'off': print('\nStarting new sequence:',model_name,'with',resampling_name)
    model_prediction = []
    y_train_resampled = []  
 
    
    #Define model
    model = define_model(model_name,random_state,verbose)
    #Split train and test sets + Apply scaling when need + Apply resampling
    X_train, X_test, y_train, y_test = split_resample_sets(model_name,resampling_name,X,y,test_size,random_state,verbose)
    #Train model
    if verbose != 'off': print('\nFitting model.')
    model = model.fit(X_train, y_train)
    if verbose != 'off': print("\nFitting model processing time: --- %s seconds ---" % (time.time() - start_time))
    #Predict on trained model
    if verbose != 'off': print('\nPredicting on model.')
    model_prediction = model.predict(X_test) 
    if verbose != 'off': print("\nModel prediction processing time: --- %s seconds ---" % (time.time() - start_time))
    #Evaluate model performance   
    cr,cm, precision, recall, fbeta_score, support = model_performance(model_name,resampling_name,y_test,model_prediction,verbose)
    
    print("\nTotal processing time: --- %s seconds ---" % (time.time() - start_time))
    
    return model,cr,cm, precision, recall, fbeta_score, support



def define_model(model_name,random_state,verbose):
    """
    Instantiates and defines Classifier models.
    
    INPUT
    model_name = Str with model name
    random_state = INT random state number
    verbose = STR to switch execution log on or off
    
    OUTPUT
    Returns intantiated model according to users choice.
    """
    start_time = time.time() #Count processing time
    if verbose != 'off': print('\nInstantiating',model_name,'model.')
    rf = RandomForestClassifier(random_state = random_state,
                                class_weight='balanced'
                               )
    lr =  LogisticRegression(random_state = random_state,
                             class_weight='balanced',
                             penalty="l2", tol=0.01, solver="saga"                             
                            )
    if model_name == 'Random Forest':
        if verbose != 'off': print('\nModel ready:',rf)
        if verbose != 'off': print("Model Instatiating processing time: --- %s seconds ---" % (time.time() - start_time))
        return rf
    elif model_name == 'Logistic Regression':
        if verbose != 'off': print('\nModel ready:',lr)
        if verbose != 'off': print("Model Instatiating processing time: --- %s seconds ---" % (time.time() - start_time))
        return lr
    else:
        print('\nNo compatible model.')


    
    
def split_resample_sets(model_name,resampling_name,X,y,test_size,random_state,verbose):
    """
    Splits and resamples X dataset and y vector.
    
    INPUT
    model_name = Str with model name
    resampling_name = Str with resampling method
    X = DataFrame with independent variables
    y = Vector with dependent (response) variable
    test_size = Float (0 to 1) for test size porcentage of the train/test split
    random_state = INT random state number
    verbose = STR to switch execution log on or off
    
    OUTPUT
    X_train = DataFrame with independent variables splitted for train set
    X_test = DataFrame with independent variables splitted for test set
    y_train = Vector with dependent variable splitted for train set
    y_test = Vector with dependent variable splitted for test set
    """
    start_time = time.time() #Count processing time
    if verbose != 'off': print('\nIniatialing split for train and test sets. \nAnalyzing need for variable rescaling.')
    if model_name == 'Logistic Regression':
        if verbose != 'off': print('\nLogistic Regression requies scaling. \nVariable rescaling necessary.')
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        X_scaled

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size = test_size, random_state = random_state)
    elif model_name == 'Random Forest':
        if verbose != 'off': print('\nRandom Forest does not require rescalling.')
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = test_size, random_state = random_state)
    
    if verbose != 'off': print('\nApplying resampling technique choosen.')
    if resampling_name == 'Baseline':
        if verbose != 'off': print("\nBaseline doesn't require resampling.")
    elif resampling_name == 'Random Over Sampling':
        if verbose != 'off': print('\nApplying',resampling_name)
        resampler = RandomOverSampler(random_state=random_state)
        X_train, y_train= resampler.fit_resample(X_train, y_train)
    elif resampling_name == 'SMOTE':
        if verbose != 'off': print('\nApplying',resampling_name)
        resampler = SMOTE(random_state=random_state)
        X_train, y_train= resampler.fit_resample(X_train, y_train)
    elif resampling_name == 'NearMiss KNN':
        if verbose != 'off': print('\nApplying',resampling_name)
        resampler = NearMiss(version=3,random_state=random_state)
        X_train, y_train= resampler.fit_resample(X_train, y_train)
    elif resampling_name == 'Random Under Sampling':
        if verbose != 'off': print('\nApplying',resampling_name)
        resampler = RandomUnderSampler(random_state=random_state)
        X_train, y_train= resampler.fit_resample(X_train, y_train)
    
    if verbose != 'off': print("\nResampling processing time: --- %s seconds ---" % (time.time() - start_time))
    return X_train, X_test, y_train, y_test



def model_performance(model_name,resampling_name,y_test,model_prediction,verbose):
    """
    Prints the performance reports: Classification report and Confusion Matrix
    
    Inputs
    model_name = Str with model name
    resampling_name = Str with resampling method
    y_test = Test vector
    model_prediction = Prediction vector
    c =
    cm = 
    precision = 
    recall = 
    fbeta_score =
    support = INT Number of 
    verbose = STR to switch execution log on or off
    
    Returns print with the reports
    """
    start_time = time.time() #Count processing time
    cr = classification_report(y_test, model_prediction)
    cm = confusion_matrix(y_test, model_prediction)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, model_prediction)
    print('\n',model_name,'with',resampling_name,' Classification Report:')
    print(cr)
    print(cm)
    
    if verbose != 'off': print("\nModel Performane processing time: --- %s seconds ---" % (time.time() - start_time))
    return cr,cm, precision, recall, fbeta_score, support
    
    # #Precision-Recall Curve gives us the correct accuracy in this imbalanced dataset case. We can see that we have a very poor accuracy for the model.
    # precision, recall, thresholds = precision_recall_curve(model_prediction, y_test)

    # # create plot
    # plt.plot(precision, recall, label='Precision-recall curve')
    # plt.xlabel('Precision')
    # plt.ylabel('Recall')
    # plt.title('Precision-recall curve')
    # plt.legend(loc="lower left")


        


def Model_Coef_Table(models, X):
    """
    Function for analyzing the different importance coefients of both Logistic Regression and Random Forest models.
    Prints out a table and bar graph of the top 10 coeficients (simple summ)
    
    INPUT
    model_prediction = List containing the outputs of model_predict function:
                            model = Class Fitted Classfier model instances (Logistic Regression or Ranfom Forest)
                            cr = Str Classification Report
                            cm = NP.Array Confusion Matrix
                            precision =
                            recall = 
                            fbeta-score = 
                            support = 
                            model_name = Str with model name
                            resampling_name = Str with resampling technique name
    X = pd.DataFrame with predictor (independent) variables  
    
    OUTPUT
    Model_Coef_Table = Pandas Dataframe with Features and Coefients
                            
    """
    Model_Coef_Table = pd.DataFrame()
    Model_Coef_Table['Features'] = X.columns
    
#     Model_Score_Table['Model'] = model_prediction[7] + model_prediction[8]
    
    for model in models:
        
        model_ = model[0] # Model
        model_name = model[1]# model_name
        resampler_name = model[2]# resampler_name
        
        print(model_)
        
        if model_name == 'Random Forest': Model_Coef_Table['Coef'+model_name+resampler_name] = model_.feature_importances_
        elif model_name == 'Logistic Regression': Model_Coef_Table['Coef'+model_name+resampler_name] = model_.coef_[0]    
       
    return Model_Coef_Table

