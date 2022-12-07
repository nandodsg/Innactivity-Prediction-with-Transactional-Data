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
from sklearn.metrics import precision_recall_curve, accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
# Oversampling and under sampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter
# Processing time
import time
# Statistic
import statistics as s
# A little try for Reloading Utils when we update
from importlib import reload  # Python 3.4+ :: Just call the u = u.reload(u) on the main notebook
import sys 

def model_run(models,resamplers,X,y,X_bbt,y_bbt,random_state,test_size,verbose,print_report):
    """
    
    INPUT
    models = Pandas List with model names
    resamples = Pandas List with resampler names
    X = DataFrame with independent variables
    y = Vector with dependent (response) variable
    random_state = INT random state number
    verbose = STR to switch execution log on or off
    print_report = STR to switch result printing on or off
    
    OUTPUT
    """
    
    start_time = time.time() #Start processing time
    print('Model training and evaluation routine started.')
    
    #Create list to hold firs run's trained models
    model_prediction = []
    
    #Collect dataframe to collect scores during first run
    Scores = ['Model',
            'Model Name',
            'Resampler Name',
            'TN',
            'FP',
            'FN',
            'TP',
            'Precision 0',
            'Precision 1',
            'Recall 0',
            'Recall 1',
            'F1-Score 0',
            'F1-Score 1',
            'Support 0',
            'Support 1',
            'FPR',
            'TPR',
            'Thresholds']
    model_scores_table = pd.DataFrame()
    model_scores_table['Scores'] = Scores
    
    #Create Dataframe to collect first run model's coeficient and feature importance measures
    model_coef_table = pd.DataFrame()
    model_coef_table['Features'] = X.columns
        
    for model_name in models:
        for resampler_name in resamplers:

            #Split, train and predict on models
            model,cr,cm, precision, recall, fbeta_score, support, fpr, tpr, thresholds =  model_predict(model_name,
                                                                                                         resampler_name,
                                                                                                         X,
                                                                                                         y,
                                                                                                         random_state,
                                                                                                         test_size,
                                                                                                         verbose,
                                                                                                         print_report)
            model_prediction.append((model,model_name,resampler_name))
            if verbose != 'off':print(model_name+' '+resampler_name+' Split, Training and Test Prediction ready.')

            #Collect Scores
            models_scores_append = [model,
                                    model_name,
                                    resampler_name,
                                    cm[0][0],
                                    cm[0][1],
                                    cm[1][0],
                                    cm[1][1],
                                    precision[0],
                                    precision[1],
                                    recall[0],
                                    recall[1],
                                    fbeta_score[0],
                                    fbeta_score[1],
                                    support[0],
                                    support[1],
                                    fpr,
                                    tpr,
                                    thresholds]

            model_scores_table[model_name+' '+resampler_name] = models_scores_append
            if verbose != 'off':print(model_name+' '+resampler_name+' Scores collected.')


            #Collect Coeficients and Feature Importance
            if model_name == 'Random Forest': model_coef_table_append = model.feature_importances_
            elif model_name == 'Logistic Regression': model_coef_table_append = model.coef_[0] 

            model_coef_table[model_name+' '+resampler_name] = model_coef_table_append
            if verbose != 'off':print(model_name+' '+resampler_name+' Coeficients and Features Importance collected.')
                
            print('           '+model_name+' '+resampler_name+' routine finished')
    
            
    # Let's transpose the Model Scores Table to get a better look
    model_scores_table = transpose_model_scores(model_scores_table)
    
    print('Model training and evaluation finished.')
    print('---------------------------------------')


    # ---------------------------------------------------------------------------------
    #Run Big Blind Test routines:
    print('Starting Big Blind Test routine.')
    
    #Create dataframe to collect Big Blind Test performance scores
    BBT_model_scores_table = pd.DataFrame()
    BBT_model_scores_table['Scores'] = Scores
    
    #Create scaled version of X to use on Logistic Regression Model
    scaler = StandardScaler().fit(X_bbt)
    X_bbt_scaled = scaler.transform(X_bbt)
    X_bbt_scaled
        
    for model in model_prediction:
        BBT_model = model[0]
        BBT_model_name = model[1]
        BBT_resampler_name = model[2]
              
        #Predict on new Big Blind Test data
        if BBT_model_name == 'Logistic Regression': 
            bbt_prediction = model[0].predict(X_bbt_scaled) 
            bbt_model_proba = model[0].predict_proba(X_bbt_scaled)
        elif BBT_model_name == 'Random Forest':
            bbt_prediction = model[0].predict(X_bbt) 
            bbt_model_proba = model[0].predict_proba(X_bbt)
        if verbose != 'off':print(BBT_model_name+' '+BBT_resampler_name+' Big Blind Data prediction ready.')

        #Evaluate perfomance of traind models on the new Big Blind Test data 
        
        cr,cm, precision, recall, fbeta_score, support, fpr, tpr, thresholds = model_performance('Big Blind Test',
                                                                                                  BBT_model_name,
                                                                                                  BBT_resampler_name,
                                                                                                  y_bbt,
                                                                                                  bbt_prediction,
                                                                                                  bbt_model_proba,
                                                                                                  verbose,
                                                                                                  print_report)
        if verbose != 'off':print(model_name+' '+resampler_name+' Big Blind Test Score evaluation finished')

        #Collect Scores
        BBT_model_scores_table_append = [BBT_model,
                                         BBT_model_name,
                                         BBT_resampler_name,
                                         cm[0][0],
                                         cm[0][1],
                                         cm[1][0],
                                         cm[1][1],
                                         precision[0],
                                         precision[1],
                                         recall[0],
                                         recall[1],
                                         fbeta_score[0],
                                         fbeta_score[1],
                                         support[0],
                                         support[1],
                                         fpr,
                                         tpr,
                                         thresholds]

        BBT_model_scores_table[BBT_model_name+' '+BBT_resampler_name] = BBT_model_scores_table_append
        if verbose != 'off':print(BBT_model_name+' '+BBT_resampler_name+' Big Blind Data Scores collected.')
            
        print('           Big Blind Test '+BBT_model_name+' '+BBT_resampler_name+' routine finished')
        
    # Let's transpose the BBT Model Scores Table to get a better look
    BBT_model_scores_table = transpose_model_scores(BBT_model_scores_table)

    print("\nTotal Run processing time: --- %s seconds ---" % (time.time() - start_time))
    
    return model_prediction, model_scores_table, model_coef_table, BBT_model_scores_table

def model_predict(model_name,resampling_name,X,y,random_state,test_size,verbose,print_report):
    """
    Trains and Fits Models with different Resampling Techniques, than predicts, scores and measures coeficient and feature importance.
    
    INPUT
    model_name = Str with model name
    resampling_name = Str with resampling technique name
    X = DataFrame with independent variables
    y = Vector with dependent (response) variable
    random_state = INT random state number
    verbose = STR to switch execution log on or off
    print_report = STR to switch result printing on or off
    
    OUTPUT
    model_prediction = Vector with chosen model with resampling predictions on the test set
    """
    start_time = time.time() #Count processing time
    if verbose != 'off':print('\n--------------------------------------------------------------------------------\n')
    if verbose != 'off': print('\nStarting new sequence:',model_name,'with',resampling_name)
    model_prediction = []
    y_train_resampled = []  
 
    
    #Define model
    model = define_model(model_name,random_state,verbose)
    #Split train and test sets + Apply scaling when need + Apply resampling
    X_train, X_test, y_train, y_test = split_resample_sets(model_name,
                                                           resampling_name,
                                                           X,
                                                           y,
                                                           test_size,
                                                           random_state,
                                                           verbose)
    #Train model
    if verbose != 'off': print('\nFitting model.')
    model = model.fit(X_train, y_train)
    if verbose != 'off': print("\nFitting model processing time: --- %s seconds ---" % (time.time() - start_time))
    #Predict on trained model
    if verbose != 'off': print('\nPredicting on model.')
    model_prediction = model.predict(X_test) 
    if verbose != 'off': print("\nModel prediction processing time: --- %s seconds ---" % (time.time() - start_time))
    #Evaluate model performance   
    model_proba = model.predict_proba(X_test)[:, 1]
    cr,cm, precision, recall, fbeta_score, support, fpr, tpr, thresholds = model_performance('First Run',
                                                                       model_name,
                                                                       resampling_name,
                                                                       y_test,
                                                                       model_prediction,
                                                                       model_proba,
                                                                       verbose,
                                                                       print_report)
    
    if verbose != 'off': print("\nTotal Model processing time: --- %s seconds ---" % (time.time() - start_time))
    
    #pyplot.show()
    
    return model,cr,cm, precision, recall, fbeta_score, support, fpr, tpr, thresholds



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

    if model_name == 'Random Forest':
        model = RandomForestClassifier(random_state = random_state,
#                                        class_weight='balanced_subsample',
#                                        max_depth = 2,
#                                        n_estimators=40,
#                                        max_leaf_nodes=10
                                       )
        if verbose != 'off': print('\nModel ready:',model)
        if verbose != 'off': print("Model Instatiating processing time: --- %s seconds ---" % (time.time() - start_time))

    elif model_name == 'Logistic Regression':
        model =  LogisticRegression(random_state = random_state,
#                                      class_weight='balanced',
#                                      penalty="l2", # l1_ratio = 0.5, only necessary with penalty="elasticnet"
#                                      tol=0.01,
#                                      solver="lbfgs",
#                                      C=0.01
                                    )
        if verbose != 'off': print('\nModel ready:',lr)
        if verbose != 'off': print("Model Instatiating processing time: --- %s seconds ---" % (time.time() - start_time))

    else:
        print('\nNo compatible model.')

    return model    
    
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
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, 
                                                            y,test_size = test_size,
                                                            random_state = random_state)
    elif model_name == 'Random Forest':
        
        if verbose != 'off': print('\nRandom Forest does not require rescalling.')
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size = test_size,
                                                            random_state = random_state)
    
    
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



def model_performance(run,model_name,resampling_name,y_test,model_prediction,model_proba,verbose,print_report):
    """
    Prints the performance reports: Classification report and Confusion Matrix
    
    Inputs
    run = Str indicating which run is being executed. Accepts only 'First Run' and 'Big Blind Test'
    model_name = Str with model name
    resampling_name = Str with resampling method
    y_test = Test vector
    model_prediction = Prediction vector
    verbose = STR to switch execution log on or off
    print_report = STR to switch resulting printing on or off
    
    Returns print with the reports
    """
    start_time = time.time() #Count processing time
    cr = classification_report(y_test, model_prediction,zero_division=0)
    cm = confusion_matrix(y_test, model_prediction)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, model_prediction)
    if print_report != 'off': print('\n',model_name,'with',resampling_name,' Classification Report:')
    if print_report != 'off': print(cr)
    if print_report != 'off': print(cm)
    
    if verbose != 'off': print("\nModel Performane processing time: --- %s seconds ---" % (time.time() - start_time))
        
    # Plot the ROC Curve    
    #Implementation derived from :
    ## https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
    ## https://stackoverflow.com/questions/33208897/how-to-interpret-this-triangular-shape-roc-auc-curve
      
    if run == 'First Run':
        fpr, tpr, thresholds = roc_curve(y_test, model_proba)
        pyplot.plot(fpr, tpr,label=run+' '+model_name+' '+resampling_name)
    elif run == 'Big Blind Test':
        fpr, tpr, thresholds = roc_curve(y_test, model_proba[:, 1])
        pyplot.plot(fpr, tpr,label=run+' '+model_name+' '+resampling_name,linestyle='--')
    pyplot.title('ROC and AUC Plot for tested models')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend(bbox_to_anchor=(1,1), loc="upper left")
    
    return cr,cm, precision, recall, fbeta_score, support, fpr, tpr, thresholds
    

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
        
        if model_name == 'Random Forest': Model_Coef_Table['Feature Importance '+model_name+resampler_name] = model_.feature_importances_
        elif model_name == 'Logistic Regression': Model_Coef_Table['Coeficient '+model_name+resampler_name] = model_.coef_[0]    
       
    return Model_Coef_Table

def transpose_model_scores(model_scores_table):
    """
    Transposes Model Scores Table for easier analysis
    
    INPUT
    model_scores_table = pd.DataFrame() with model scores
    
    OUTPUT
    model_scores_table_T = pd.DataFrame() of transposed model scores table
    """
    # Let's transpose the Model Scores Table to get a better look
    model_scores_table_T = model_scores_table.T
    # Now let's promote the first row as header and drop the index
    model_scores_table_T = model_scores_table_T.rename(columns=model_scores_table_T.iloc[0]).drop(model_scores_table_T.index[0]).reset_index(drop=True)
    # Let's clean out the score related to predicting the majority class (1) and focus only on the minority
    #model_scores_table_T = model_scores_table_T.drop(columns=['Precision 1','Recall 1','F1-Score 1'])
    return model_scores_table_T
