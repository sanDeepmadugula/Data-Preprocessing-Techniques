#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
os.chdir('D:/python using jupyter/Data Preprocessing')


# In[5]:


df_card = pd.read_csv('creditcard.csv')


# In[7]:


df_card.head()


# In[9]:


df_card[['Time','Amount']].describe()


# In[10]:


df_card.info()


# In[11]:


df_card['Class'].value_counts()


# In[12]:


print('Fraudulent transactions account for {:.2f}% of the dataset'.format(df_card['Class'].value_counts()[1]/len(df_card)*100))


# In[13]:


df_card[['Amount','Class']].groupby('Class').mean()


# We can observe that fraud is having slightly higher mean

# # Even though regular transactions have a higher transaction amount
# 

# In[14]:


df_card[['Amount','Class']].groupby('Class').max()


# In[15]:


df_card.head()


# In[16]:


df_card['Amount'].value_counts()


# Sorting the fraudulent transactions by Amount

# In[17]:


df_card[df_card['Class']==1][['Amount','Class']].sort_values(by='Amount',ascending=False).head(10)


# In[18]:


df_card[df_card['Class']==1][['Amount','Class']]['Amount'].value_counts()


# # Exploratory Data Analysis

# In[19]:


def get_transaction_average():
    # Fraud detection mean
    fradulent_transaction_mean = df_card[df_card['Class']==1]['Amount'].mean()
    # regular transaction mean
    normal_transaction_mean   = df_card[df_card['Class']==0]['Amount'].mean()
    # creating an array with the mean values
    return[fradulent_transaction_mean,normal_transaction_mean]


# In[20]:


# Get the mean value of each transaction type
mean_arr = get_transaction_average()
# calculate overall mean
overall_mean = df_card['Amount'].mean()


# In[21]:


overall_mean


# In[24]:


fig = plt.figure(figsize=(10, 8))
## Labels to replace the elements' indexes in the x-axis
xticks_labels = ['Fraudulent transactions', 'Regular transactions']
## X-axis elements
xticks_elements = [item for item in range(0,len(mean_arr))]
ax = plt.gca()
## Plot the bar char custom bar colors
plt.bar(xticks_elements, mean_arr, color='#2F4F4F')
## Map the xticks to their string descriptions, then rotate them to make them more readable
plt.xticks(xticks_elements, xticks_labels, rotation=70)
## Draw a horizontal line to show the overall mean to compare with each category's mean
plt.axhline(overall_mean, color='#e50000', animated=True, linestyle='--')
## Annotate the line to explain its purpose
ax.annotate('Overall Mean', xy=(0.5, overall_mean), xytext=(0.5, 110),
            arrowprops=dict(facecolor='#e50000', shrink=0.05))
## Set the x-axis label
plt.xlabel('Transactions')
## Set the y-axis label
plt.ylabel('Average amount in $ Dollar')
## Show the plot
plt.show()


# In[26]:


# Describing the amount values for the fraulent transactions
describe_arr = df_card[df_card['Class']==1]['Amount'].describe()
describe_arr


# In[27]:


# lets create a new figure
plt.figure(figsize=(10,8))
# Filter out the fraudent transaction from dataframe
df_fraudulent = df_card[df_card['Class']==1]
sns.boxplot(x="Class", y="Amount", 
                 data=df_fraudulent, palette='muted')


# In[28]:


# lets create a new figure
plt.figure(figsize=(10,8))
# Filter out the fraudent transaction from dataframe
df_fraudulent = df_card[df_card['Class']==0]
sns.boxplot(x="Class", y="Amount", 
                 data=df_fraudulent, palette='muted')


# In[29]:


## Creates a new figure 
plt.figure(figsize=(10, 8))
# draw distribution plot (histogram) from amount values
sns.distplot(df_card['Amount'],kde=True,hist=True,norm_hist=True)


# In[30]:


## Creates a new figure 
plt.figure(figsize=(10, 8))
## Draw a distribution plot (histogram) from fraudulent transactions data
sns.distplot(df_fraudulent['Amount'], kde=True, hist=True, norm_hist=True)
## Check that most transactions are clustered around $0 and $500.


# In[31]:


df_card.head()


# # preparing data for model

# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


# now scale the features
sc = StandardScaler()
df_card['scaled_amount'] = sc.fit_transform(df_card.iloc[:,29].values.reshape(-1,1))
# now drop the old one
df_card.drop('Amount',axis=1,inplace=True)


# In[34]:


df_card.head(5)


# In[35]:


## Set the features to the X variable
X = df_card.drop(['Time','Class'],axis=1)
X


# In[36]:


y_target = df_card['Class']


# In[37]:


y_target.head()


# # Models

# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc,roc_auc_score,average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# # Model Utility Function

# In[42]:


X_train,X_test,y_train,y_test = train_test_split(X,y_target,random_state=42)


# In[43]:


## This is a generic function to calculate the auc score which is used several times in this notebook
def evaluate_model_auc(model,X_test_parameter,y_test_parameter):
    y_pred = model.predict(X_test_parameter)
        ## False positive rate, true positive rate and treshold
    fp_rate,tp_rate,treshold = roc_curve(y_test_parameter,y_pred)
    ## calculate the auc score
    auc_score = auc(fp_rate,tp_rate)
    return (auc_score)
    


# In[44]:


## This is a generic function to plot the area under the curve (AUC) for a model
def plot_auc(model,X_test,y_test):
    ##predict
    y_pred = model_predict(X_test)
    
    ##Calculate the auc score
    fp_rate,tp_rate,treshold = roc_curve(y_test,y_pred)
    auc_score = auc(fp_rate,tp_rate)
    
        ## Creates a new figure and adds its parameters
    plt.figure()
    plt.title('ROC Curve')
    ## Plot the data - false positive rate and true positive rate
    plt.plot(fp_rate, tp_rate, 'b', label = 'AUC = %0.2f' % auc_score)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
        


# In[45]:


## This is a generic utility function to calculate a model's score
def evaluate_model_score(model,X_test,y_test):
    return model.score(X_test,y_test)


# In[46]:


## This is a generic function to create a classification report and return it to the model. The target
## variables have been mapped to the transaction types
def evaluate_classification_report(model, y_test):
    return classification_report(y_test, model.predict(X_test), target_names=['Regular transaction',
                                                                      'Fraudulent transaction'])


# In[47]:


## This utility function evaluates a model using some common metrics such as accurary and auc. Also, it
## prints out the classification report for the specific model
def evaluate_model(model_param, X_test_param, y_test_param):
    print("Model evaluation")
    print("Accuracy: {:.5f}".format(evaluate_model_score(model_param, X_test_param, y_test_param)))
    print("AUC: {:.5f}".format(evaluate_model_auc(model_param, X_test_param, y_test_param)))
    print("\n#### Classification Report ####\n")
    print(evaluate_classification_report(model_param, y_test_param))
    plot_auc(model_param, X_test_param, y_test_param)


# In[48]:


## This is a shared function used to print out the results of a gridsearch process
def gridsearch_results(gridsearch_model):
    print('Best score: {} '.format(gridsearch_model.best_score_))
    print('\n#### Best params ####\n')
    print(gridsearch_model.best_params_)


# In[49]:


# Returns the Random Forest model which the n_estimators returns the highest score in order to improve 
# the results of the default classifier
# min_estimator - min number of estimators to run
# max_estimator - max number of estimators to run
# X_train, y_train, X_test, y_test - splitted dataset
# scoring function: accuracy or auc
def model_selection(min_estimator, max_estimator, X_train_param, y_train_param,
                   X_test_param, y_test_param, scoring='accuracy'):
    scores = [] 
    ## Returns the classifier with highest accuracy score
    if (scoring == 'accuracy'):
        for n in range(min_estimator, max_estimator):
            rfc_selection = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_train_param, y_train_param)
            score = evaluate_model_score(rfc_selection, X_test_param, y_test_param)
            print('Number of estimators: {} - Score: {:.5f}'.format(n, score))
            scores.append((rfc_selection, score))
            
    ## Returns the classifier with highest auc score
    elif (scoring == 'auc'):
         for n in range(min_estimator, max_estimator):
            rfc_selection = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_train_param, y_train_param)
            score = evaluate_model_auc(rfc_selection, X_test_param, y_test_param)
            print('Number of estimators: {} - AUC: {:.5f}'.format(n, score))
            scores.append((rfc_selection, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)[0][0]


# Dealing with imbalance classes

# In[52]:


from imblearn.over_sampling import SMOTE
from sklearn.utils import resample


# In[53]:


from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
# Making a copy of the dataset (could've been done using df.copy())
dataset = df_card[df_card.columns[1:]]
## Defines the features to the dataset_features variable
dataset_features = dataset.drop(['Class'], axis=1)
## Defines the target feature to the dataset_target variable
dataset_target = dataset['Class']


# In[ ]:


## Split the data once again
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(dataset_features,
                                                   dataset_target,
                                                   random_state=42)


# In[ ]:


## This function generates a balanced X_train and y_train from the original dataset to fit the model
def get_balanced_train_data(df):
    sm = SMOTE(random_state=42, ratio = 1.0)
    X_train_res, y_train_res = sm.fit_sample(X_train_2, y_train_2)
    ## Returns balanced X_train & y_train
    return (X_train_res, y_train_res)


# In[ ]:


## Calling the function to get scalled training data
(X_train_resampled, y_train_resampled) = get_balanced_train_data(df_card)


# # SVM

# In[ ]:


## Creating a SVC model with default parameters
svc = svm.SVC()
svc.fit(X_train_2, y_train_2)


# In[ ]:


## Evaluating the model
evaluate_model(svc, X_test_2, y_test_2)


# Cross valiadtion using parameter tuning

# In[ ]:


## Parameters grid to be tested on the model
parameters = {
    'C': [1, 5, 10, 15],
    'degree':[1, 2, 3, 5],
    'kernel': ['linear'],
    'class_weight': ['balanced', {0:1, 1:10}, {0:1, 1:15}, {0:1, 1:20}],
    'gamma': [0.01, 0.001, 0.0001, 0.00001]
    }


# In[ ]:


## Creates a gridsearch to find the best parameters for this dataset.
clf = GridSearchCV(estimator=svm.SVC(random_state=42),
                   ## Passes the parameter grid as argument (these parameters will be tested
                   ## when this model is created)
                   param_grid=parameters,
                   ## Run the processes in all CPU cores
                   n_jobs=-1,
                   ## Set the scoring method to 'roc_auc'
                   scoring='roc_auc')


# In[ ]:


## Fit the gridsearch model to the data
# clf.fit(X_train_2[:5000], y_train_2[:5000])
## Find the model with the best score achieved and the best parameters to use
# gridsearch_results(clf)


# # Using the optimal parameter

# In[ ]:


## Creates a SVC model with the optimal parameters found in the previous step
svc_grid_search = svm.SVC(C=1,
                          kernel='linear',
                          degree=1,
                          class_weight={0:1, 1:10},
                          gamma=0.01,
                          random_state=42)
svc_grid_search.fit(X_train_2[:5000], y_train_2[:5000])


# In[ ]:


## Evaluate the model
evaluate_model(svc_grid_search, X_test_2, y_test_2)


# # RandomForest classifier

# In[ ]:


# Creates a Random Forest Classifier with default parameters
model_rfc = RandomForestClassifier().fit(X_train_2, y_train_2)
## Evaluate the model
evaluate_model(model_rfc, X_test, y_test)


# In[ ]:


## Creating a model selecting the best number of estimators
rfc_model = model_selection(5, 15, X_train, y_train, X_test, y_test, scoring='auc')


# In[ ]:


## Evaluate the model
evaluate_model(rfc_model, X_test, y_test)


# # Training with balanced dataset

# In[ ]:


## Select the model with the best number of estimators using the balanced dataset
rfc_smote = model_selection(5, 15, X_train_resampled, y_train_resampled,
                     X_test_2, y_test_2, scoring='auc')


# In[ ]:


## Evaluate the model with AUC metric
evaluate_model(rfc_smote, X_test_2, y_test_2)


# In[ ]:


## Show the most important features from the dataset
sorted(rfc_smote.feature_importances_, reverse=True)[:5]


# In[ ]:


## Itemgetter import
from operator import itemgetter


# In[ ]:


## Loading features and importance
features = [i for i in X.columns.values]
importance = [float(i) for i in rfc_smote.feature_importances_]
feature_importance = []

## Creating a list of tuples concatenating feature names and its importance
for item in range(0, len(features)):
    feature_importance.append((features[item], importance[item]))

## Sorting the list
feature_importance.sort(key=itemgetter(1), reverse=True)

## Printing the top 5 most important features
feature_importance[:5]


# # Grid search and randomForest

# In[ ]:


## Parameters to use with the RFC model
parameters_rfc = { 
    'n_estimators': [5, 6, 7, 8, 9, 10, 13, 15],
#     'class_weight': ['balanced'],
    'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}


# In[ ]:


## Gridsearch to get the best parameters for RFC
rfc_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42,
                                                               n_jobs=-1),
                               param_grid=parameters_rfc,
                               cv=10, 
                               scoring='roc_auc',
                               return_train_score=True)


# https://www.kaggle.com/luizhsda/0-94-auc-with-imbalanced-dataset/data

# In[ ]:




