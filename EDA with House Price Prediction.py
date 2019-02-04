#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
os.chdir('D:/python using jupyter/Data Preprocessing/House Price Prediction')
import seaborn as sns
sns.set(style='white',color_codes=True)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)
plt.rcParams["axes.labelsize"] = 16


# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[4]:


train.shape


# In[5]:


test.shape


# In[6]:


train.head(2)


# In[7]:


test.head(2)


# # Imputing Missing values

# In[8]:


train_and_test_data = pd.concat([train,test])
sum_result = train_and_test_data.isna().sum(axis=0).sort_values(ascending=False)
missing_values_columns = sum_result[sum_result>0]
print('They are %s columns with missing values : \n%s' % (missing_values_columns.count(), [(index, value) for (index, value) in missing_values_columns.iteritems()]))


# In[10]:


def impute_missing_values(train_data):
    dataset = train_data
    dataset["PoolQC"].fillna("NA", inplace=True)
    dataset["MiscFeature"].fillna("NA", inplace=True)
    dataset["Alley"].fillna("NA", inplace=True)
    dataset["Fence"].fillna("NA", inplace=True)
    dataset["FireplaceQu"].fillna("NA", inplace=True)
    dataset["LotFrontage"].fillna(dataset["LotFrontage"].median(), inplace=True)
    dataset["GarageType"].fillna("NA", inplace=True)
    dataset["GarageQual"].fillna("NA", inplace=True)
    dataset["GarageCond"].fillna("NA", inplace=True)
    dataset["GarageFinish"].fillna("NA", inplace=True)
    dataset["GarageYrBlt"].fillna(dataset["GarageYrBlt"].median(), inplace=True)
    dataset["BsmtExposure"].fillna("NA", inplace=True)
    dataset["BsmtFinType2"].fillna("NA", inplace=True)
    dataset["BsmtQual"].fillna("NA", inplace=True)
    dataset["BsmtCond"].fillna("NA", inplace=True)
    dataset["BsmtFinType1"].fillna("NA", inplace=True)
    dataset["MasVnrArea"].fillna(dataset["MasVnrArea"].median(), inplace=True)
    dataset["MasVnrType"].fillna("None", inplace=True)
    dataset["Electrical"].fillna("SBrkr", inplace=True)  # SBrkr is the most common value for 1334 houses
    dataset["BsmtQual"].fillna("NA", inplace=True)
    dataset["MSZoning"].fillna("TA", inplace=True)
    dataset["BsmtFullBath"].fillna(0, inplace=True)
    dataset["BsmtHalfBath"].fillna(0, inplace=True)
    dataset["Utilities"].fillna("AllPub", inplace=True)
    dataset["Functional"].fillna("Typ", inplace=True)
    dataset["Electrical"].fillna("SBrkr", inplace=True)
    dataset["Exterior2nd"].fillna("VinylSd", inplace=True)
    dataset["KitchenQual"].fillna("TA", inplace=True)
    dataset["Exterior1st"].fillna("VinylSd", inplace=True)
    dataset["GarageCars"].fillna(0, inplace=True)
    dataset["GarageArea"].fillna(dataset["GarageArea"].median(), inplace=True)
    dataset["TotalBsmtSF"].fillna(dataset["TotalBsmtSF"].median(), inplace=True)
    dataset["BsmtUnfSF"].fillna(dataset["BsmtUnfSF"].median(), inplace=True)
    dataset["BsmtFinSF2"].fillna(dataset["BsmtFinSF2"].median(), inplace=True)
    dataset["BsmtFinSF1"].fillna(dataset["BsmtFinSF1"].median(), inplace=True)
    dataset["SaleType"].fillna("WD", inplace=True) 
    return dataset

train_data = impute_missing_values(train)
test_data  = impute_missing_values(test)


# # Data Exploration

# In[12]:


plt.hist(train_data['SalePrice'],50)
plt.xlabel('SalePrice')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[13]:


train_data['SalePrice'].describe()


# # Important numerical varaibles

# In[15]:


numeric_cols = list(train_data.select_dtypes(include=[np.number]))
numeric_cols.remove('Id')
numeric_cols.remove('MSSubClass')
print("Here are the %s numeric variables : \n %s" % (len(numeric_cols), numeric_cols))


# In[17]:


numerical_values = train_data[list(train_data.select_dtypes(include=[np.number]))]
# Get the more correlated variables by sorting in descending order for the SalePrice column
ix = numerical_values.corr().sort_values('SalePrice',ascending=False).index
df_sorted_by_correaltion = numerical_values.loc[:,ix]
# take only the first 15 more correlated variables
fifteen_more_correlated = df_sorted_by_correaltion.iloc[:,:15]
corr = fifteen_more_correlated.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style('white'):
    ax = sns.heatmap(corr,mask=mask,annot=True)


# In[18]:


sns.boxplot(x="OverallQual", y="SalePrice", data=train_data[['OverallQual', 'SalePrice']])
plt.show()


# In[19]:


sns.pairplot(data=train_data, x_vars=['GrLivArea'], y_vars=['SalePrice'], size=9, kind='reg')
plt.show()


# # Important Categorical Variables : MSSubClass and Neighborhood

# In[20]:


train_data['MSSubClass'].head(3)


# In[21]:


train_data['Neighborhood'].head(3)


# In[22]:


train_data.replace({'MSSubClass': {
    20: "1-STORY 1946 & NEWER ALL STYLES",
    30: "1-STORY 1945 & OLDER",
    40: "1-STORY W/FINISHED ATTIC ALL AGES",
    45: "1-1/2 STORY - UNFINISHED ALL AGES",
    50: "1-1/2 STORY FINISHED ALL AGES",
    60: "2-STORY 1946 & NEWER",
    70: "2-STORY 1945 & OLDER",
    75: "2-1/2 STORY ALL AGES",
    80: "SPLIT OR MULTI-LEVEL",
    85: "SPLIT FOYER",
    90: "DUPLEX - ALL STYLES AND AGES",
   120: "1-STORY PUD (Planned Unit Development) - 1946 & NEWER",
   150: "1-1/2 STORY PUD - ALL AGES",
   160: "2-STORY PUD - 1946 & NEWER",
   180: "PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",
   190: "2 FAMILY CONVERSION - ALL STYLES AND AGES",
}}, inplace=True)


# In[23]:


train_data['MSSubClass'].head(3)


# In[25]:


g = sns.boxplot(x="MSSubClass", y="SalePrice", data=train_data)

for item in g.get_xticklabels():
    item.set_rotation(75)


# In[29]:


neighborhood_median_plot = train_data.groupby('Neighborhood')['SalePrice'].median().plot(kind='bar')
neighborhood_median_plot.set_ylabel('SalePrice')
h = neighborhood_median_plot.axhline(train_data['SalePrice'].mean())


# In[30]:


count_neighborhood_plot = train_data.groupby('Neighborhood')['Neighborhood'].count().plot(kind='bar')
count_neighborhood_plot = count_neighborhood_plot.set_ylabel('Count')


# # Other Important variables : Garage variables¶
# 

# In[31]:


fig, axes = plt.subplots(nrows=2, ncols=3, squeeze=True)
figsize = (15, 10)
train_data.groupby("GarageType")["GarageType"].count().plot(kind='bar', ax=axes[0][0], figsize=figsize).set_ylabel('Count')
train_data.groupby("GarageFinish")["GarageFinish"].count().plot(kind='bar', ax=axes[0][1], figsize=figsize).set_ylabel('Count')
train_data.groupby("GarageCars")["GarageCars"].count().plot(kind='bar', ax=axes[0][2], figsize=figsize).set_ylabel('Count')
train_data.groupby("GarageQual")["GarageQual"].count().plot(kind='bar', ax=axes[1][0], figsize=figsize).set_ylabel('Count')
train_data.groupby("GarageCond")["GarageCond"].count().plot(kind='bar', ax=axes[1][1], figsize=figsize).set_ylabel('Count')
train_data.groupby("GarageYrBlt")["GarageYrBlt"].count().plot(kind='line', ax=axes[1][2], figsize=figsize).set_ylabel('Count')
fig.tight_layout(pad=3, w_pad=3, h_pad=3)


# In[32]:


garage_cars_median_plot = train_data.groupby('GarageCars')['SalePrice'].median().plot(kind='bar')
garage_cars_median_plot.set_ylabel('SalePrice')
h = garage_cars_median_plot.axhline(train_data['SalePrice'].mean())


# In[33]:


g = sns.pairplot(data=train_data, x_vars=['GarageArea'], y_vars=['SalePrice'], size=6, kind='reg')


# # Feature Map

# In[34]:


def transform_variables(dataset):
    copy = dataset
    copy.replace({'MSSubClass': {
        20: "1-STORY 1946 & NEWER ALL STYLES",
        30: "1-STORY 1945 & OLDER",
        40: "1-STORY W/FINISHED ATTIC ALL AGES",
        45: "1-1/2 STORY - UNFINISHED ALL AGES",
        50: "1-1/2 STORY FINISHED ALL AGES",
        60: "2-STORY 1946 & NEWER",
        70: "2-STORY 1945 & OLDER",
        75: "2-1/2 STORY ALL AGES",
        80: "SPLIT OR MULTI-LEVEL",
        85: "SPLIT FOYER",
        90: "DUPLEX - ALL STYLES AND AGES",
       120: "1-STORY PUD (Planned Unit Development) - 1946 & NEWER",
       150: "1-1/2 STORY PUD - ALL AGES",
       160: "2-STORY PUD - 1946 & NEWER",
       180: "PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",
       190: "2 FAMILY CONVERSION - ALL STYLES AND AGES",
    }}, inplace=True)
    # one hot encoding
    one_hot_columns = [
        'Neighborhood', 'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Heating',
        'Electrical', 'Functional', 'GarageType', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'Foundation'
    ]
    for col_name in one_hot_columns:
        copy = pd.concat([copy, pd.get_dummies(copy[col_name], prefix=col_name)], axis=1)
        copy = copy.drop(col_name, axis=1)
        
    # ordinal variables transformation
    quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
    basement_map = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    ordinal_maps = {
        "ExterCond": quality_map,
        "ExterQual": quality_map,
        "LandSlope": {'Gtl': 0, 'Mod': 1, 'Sev': 2},
        "MasVnrType": {'None': 0, 'BrkCmn': 0, 'BrkFace': 1, 'Stone': 2},
        "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0},
        "BsmtFinType1": basement_map,
        "BsmtFinType2": basement_map,
        "BsmtQual": quality_map,
        "BsmtCond": quality_map,
        "HeatingQC": quality_map,
        "CentralAir": {'N': 0, 'Y': 1},
        "KitchenQual": quality_map,
        "FireplaceQu": quality_map,
        "GarageQual": quality_map,
        "GarageCond": quality_map,
        "GarageFinish": {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
        "PavedDrive": {'N': 0, 'P': 1, 'Y': 2},
        "PoolQC": quality_map
    }
    for col_name, matching_map in ordinal_maps.items():
        copy[col_name] = copy[col_name].replace(matching_map)
    
    # remove high correlated variables to other variables
    copy = copy.drop(['YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'TotalBsmtSF', 'TotRmsAbvGrd', 'BsmtFinSF1'], axis=1)
    return copy


# In[35]:


result = pd.DataFrame()
result['ExterQual'] = train_data['ExterQual'].replace({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
result.head()


# In[36]:


garage_cars_median_plot = train_data.groupby('MasVnrType')['SalePrice'].median().plot(kind='bar')
garage_cars_median_plot.set_ylabel('SalePrice')
h = garage_cars_median_plot.axhline(train_data['SalePrice'].mean())


# In[37]:


result = pd.DataFrame()
result['MasVnrType'] = train_data['MasVnrType'].replace({'None': 0, 'BrkCmn': 0, 'BrkFace': 1, 'Stone': 2})
result.head(n=8)


# In[38]:


result = pd.get_dummies(train_data['LotConfig'],prefix='LotConfig')
result.head()


# # MOdelling

# In[40]:


y_train = train_data['SalePrice']
X_train = transform_variables(train_data)

X_test = test_data
impute_missing_values(X_test)
X_test = transform_variables(X_test)

for col_name in list(X_train.columns):
    if col_name not in X_test.columns:
        X_test[col_name] = 0

# need to investigate why X_test got extra columns compare to X_train
test_cols = list(X_test.columns)
train_cols = list(X_train.columns)
def Diff(li1, li2):
    return (list(set(li1) - set(li2)))
X_test = X_test.drop(Diff(test_cols, train_cols), axis=1)        

predictor_cols = [col for col in X_train 
                  if col != 'SalePrice'
                 ]

print(str(X_test.shape) + " should be similar to " + str(X_train.shape))


# # Lasso Regression Model
# 

# In[41]:


from math import floor
from sklearn.preprocessing import MinMaxScaler
        
# filter some variables under represented
# number_of_ones_by_cols = X_train.astype(bool).sum(axis=0)
# less_than_ten_ones_cols = number_of_ones_by_cols[number_of_ones_by_cols < 10].keys()
# X_train = X_train.drop(list(less_than_ten_ones_cols), axis=1)
# X_test = X_test.drop(list(less_than_ten_ones_cols), axis=1)
        
# cols_to_scale = ['GrLivArea', 'TotalBsmtSF', 'GarageArea']
# scaler = MinMaxScaler()
# X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
# X_test[cols_to_scale] = scaler.fit_transform(X_test[cols_to_scale])

from sklearn import linear_model

clf = linear_model.Lasso(alpha=1, max_iter=10000)
clf.fit(X_train[predictor_cols], y_train)

y_pred = clf.predict(X_test[predictor_cols])

print(clf.intercept_)
print(y_pred)
my_submission = pd.DataFrame({'Id': X_test.Id, 'SalePrice': y_pred})
my_submission.to_csv('lasso.csv', index=False)


# In[ ]:




