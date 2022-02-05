#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
# get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[8]:


#loading the dataset
df=pd.read_csv(r'C:\Users\somi syed\.jupyter\dataset\loan_prediction.csv')
df.head()


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


# find the null values
df.isnull().sum()


# In[12]:


# fill the missing values for numerical terms - mean
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())


# In[13]:


# fill the missing values for categorical terms - mode
df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])


# In[14]:


df.isnull().sum()


# In[15]:


# categorical attributes visualization
sns.countplot(df['Gender'])


# In[16]:


sns.countplot(df['Married'])


# In[17]:


sns.countplot(df['Dependents'])


# In[18]:


sns.countplot(df['Education'])


# In[19]:


sns.countplot(df['Self_Employed'])


# In[20]:


sns.countplot(df['Property_Area'])


# In[21]:


sns.countplot(df['Loan_Status'])


# In[22]:


# numerical attributes visualization
sns.distplot(df["ApplicantIncome"])


# In[23]:


sns.distplot(df["CoapplicantIncome"])


# In[24]:


sns.distplot(df["LoanAmount"])


# In[25]:


sns.distplot(df['Loan_Amount_Term'])


# In[26]:


sns.distplot(df['Credit_History'])


# In[27]:


# total income
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# In[28]:


# apply log transformation to the attribute
df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome']+1)
sns.distplot(df["ApplicantIncomeLog"])


# In[29]:


df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome']+1)
sns.distplot(df["CoapplicantIncomeLog"])


# In[30]:


df['LoanAmountLog'] = np.log(df['LoanAmount']+1)
sns.distplot(df["LoanAmountLog"])


# In[31]:


df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term']+1)
sns.distplot(df["Loan_Amount_Term_Log"])


# In[32]:


df['Total_Income_Log'] = np.log(df['Total_Income']+1)
sns.distplot(df["Total_Income_Log"])


# In[33]:


#Coorelation Matrix
corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap="BuPu")


# In[34]:


df.head()


# In[35]:


# drop unnecessary columns
cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Total_Income", 'Loan_ID', 'CoapplicantIncomeLog']
df = df.drop(columns=cols, axis=1)
df.head()


# In[36]:


#Label Encoding
from sklearn.preprocessing import LabelEncoder
cols = ["Gender","Married","Education","Self_Employed","Property_Area","Loan_Status","Dependents"]
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()


# In[46]:


##Train-Test Split
# specify input and output attributes
X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[43]:


##Model Training
# classify function
from sklearn.model_selection import cross_val_score
def classify(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print("Accuracy is", model.score(x_test, y_test)*100)
    # cross validation - it is used for better validation of model
    # eg: cv-5, train-4, test-1
    score = cross_val_score(model, x, y, cv=5)
    print("Cross validation is",np.mean(score)*100)


# In[47]:
#
#
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# classify(model, X, y)
#
#
# # In[48]:
#
#
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# classify(model, X, y)
#
#
# # In[49]:
#
#
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# classify(model, X, y)


# In[50]:


from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# classify(model, X, y)


# In[51]:


##Hyperparameter tuning
model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=1)
classify(model, X, y)


# In[53]:


#Confusion Matrix
model =RandomForestClassifier()
model.fit(x_train, y_train)


# In[55]:


from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[56]:


sns.heatmap(cm, annot=True)


# In[12]:

import pickle
pickle.dump(model,open("loan.pkl",'wb'))
