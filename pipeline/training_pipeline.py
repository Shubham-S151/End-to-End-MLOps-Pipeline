# Core libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Sklearn - Model Selection
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score

# Sklearn - Preprocessing
from sklearn.preprocessing import PowerTransformer, LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler

# Sklearn - Models (Classification + Regression)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

# Sklearn - Ensemble
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    BaggingClassifier, VotingClassifier, StackingClassifier,
    RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,
    BaggingRegressor, VotingRegressor, StackingRegressor
)

# Sklearn - Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error
)

# External library
from xgboost import XGBRegressor,XGBClassifier

# create a function to validate a Regression model
def regression_model_scores(model,xtrain,ytrain,xtest,ytest):
    global m,r2_test,r2_train,mse,rmse,mae,mape
    m=[]
    r2_test=[]
    r2_train=[]
    mse=[]
    rmse=[]
    mae=[]
    mape=[]
    mod=model
    mod.fit(xtrain,ytrain)
    pred_h=mod.predict(xtest)
    pred_h_train=mod.predict(xtrain)

    print('test_R2_score :',r2_score(ytest,pred_h))
    print('train_R2_score :',r2_score(ytrain,pred_h_train))
    print('mean_squared_error',mean_squared_error(ytest,pred_h))
    print('root_mean_squared_error',np.sqrt(mean_squared_error(ytest,pred_h)))
    print('mean_absolute_error',mean_absolute_error(ytest,pred_h))
    print('mean_absolute_percentage_error',mean_absolute_percentage_error(ytest,pred_h))

    response=input('Do you want to save this model? Y?N :')
    if response.lower()=='y':
        global score_card
        m.append(str(model))
        r2_test.append(r2_score(ytest,pred_h))
        r2_train.append(r2_score(ytrain,pred_h_train))
        mse.append(mean_squared_error(ytest,pred_h))
        rmse.append(np.sqrt(mean_squared_error(ytest,pred_h)))
        mae.append(mean_absolute_error(ytest,pred_h))
        mape.append(mean_absolute_percentage_error(ytest,pred_h))

        score_card=pd.DataFrame({'Model':m,
                                 'R_square_test':r2_test,
                                 'R_square_train':r2_train,
                                 'MSE':mse,
                                 'RMSE':rmse,
                                 'MAE':mae,
                                 'MAPE':mape})
        return score_card
    else:
        return score_card
    
# create a function to validate a Classification model
def classification_model_scores(model=LogisticRegression(),xtrain=None,ytrain=None,xtest=None,ytest=None):
    '''This function only applicable for **Classifiacton Problem**
    \nThis is the function to store all the scores and checking scores and seeing roc_curve'''
    global modl,accuracy,recall,prec,f1,cohen
    modl=[]
    accuracy=[]
    recall=[]
    prec=[]
    f1=[]
    cohen=[]
    mod=model
    mod.fit(xtrain,ytrain)
    ypreds=mod.predict_proba(xtest)[:,1]
    ypredh=mod.predict(xtest)
    print('\n')
    print('Confusion Matrix :','\n',confusion_matrix(ytest,ypredh))
    print('\n')
    print('Classification Report :','\n',classification_report(ytest,ypredh))
    print('\n')
    print(f'Cohen-Kappa Score : {cohen_kappa_score(ytest,ypredh)}')
    print(f'F1 Score : {f1_score(ytest,ypredh)}')
    print(f'Accuracy : {accuracy_score(ytest,ypredh)}')
    print(f'Recall : {recall_score(ytest,ypredh)}')
    print('\n')
    fpr,tpr,thres=roc_curve(ytest,ypreds)
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],color='red',ls='--')
    plt.title(f'ROC AUC : {round(roc_auc_score(ytest,ypreds),4)}')
    plt.show()

    q=input('Do you want to save this model(y/n):')
    if q.lower()=='y':
        global score_card
        modl.append(str(mod))
        accuracy.append(round(accuracy_score(ytest,ypredh),4))
        recall.append(round(recall_score(ytest,ypredh),4))
        prec.append(round(precision_score(ytest,ypredh),4))
        f1.append(round(f1_score(ytest,ypredh),4))
        cohen.append(round(cohen_kappa_score(ytest,ypreds),4))
        
        score_card=pd.DataFrame({'Model':modl,
                                 'Accuracy':accuracy,
                                 'Recall':recall,
                                 'Precision':prec,
                                 'F1_Score':f1,
                                 'Cohen_Kappa_Score':cohen})
        return score_card
    else :
        return score_card