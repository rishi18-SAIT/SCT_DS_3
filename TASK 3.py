#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:



df = pd.read_csv("bank-additional.csv",delimiter=';')
df.rename(columns={'y':'deposit'}, inplace=True)
df.head()


# In[4]:


df.info()


# In[5]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.dtypes


# In[10]:


df.dtypes.value_counts()


# In[11]:


df.duplicated().sum()


# In[12]:


df.isna().sum()


# In[13]:


cat_cols = df.select_dtypes(include='object').columns
print(cat_cols)

num_cols = df.select_dtypes(exclude='object').columns
print(num_cols)


# In[14]:


df.describe()


# In[15]:


df.describe(include='object')


# In[16]:


df.hist(figsize=(10,10),color='#00FFFF')
plt.show()


# In[17]:


# Calculate the number of rows and columns for subplots
num_plots = len(cat_cols)
num_rows = (num_plots + 1) // 2  # Add 1 and divide by 2 to round up for odd numbers
num_cols = 2

# Create a new figure
plt.figure(figsize=(20, 25))  # Adjust the figure size as needed

# Loop through each feature and create a countplot
for i, feature in enumerate(cat_cols, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.countplot(x=feature, data=df, palette='Wistia')
    plt.title(f'Bar Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)

# Adjust layout to prevent overlap of subplots
plt.tight_layout()
plt.show()


# In[18]:



df.plot(kind='box', subplots=True, layout=(2,5),figsize=(20,10),color='#7b3f00')
plt.show()


# In[19]:


# Exclude non-numeric columns
numeric_df = df.drop(columns=cat_cols)

# Compute the correlation matrix
corr = numeric_df.corr()

# Print the correlation matrix
print(corr)

# Filter correlations with absolute value >= 0.90
corr = corr[abs(corr) >= 0.90]

sns.heatmap(corr,annot=True,cmap='Set3',linewidths=0.2)
plt.show()


# In[20]:


high_corr_cols = ['emp.var.rate','euribor3m','nr.employed']


# In[21]:


df1 = df.copy()
df1.columns


# In[22]:


df1.drop(high_corr_cols,inplace=True,axis=1)  # axis=1 indicates columns
df1.columns


# In[23]:


df1.shape


# In[24]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df_encoded = df1.apply(lb.fit_transform)
df_encoded


# In[25]:


df_encoded['deposit'].value_counts()


# In[26]:


x = df_encoded.drop('deposit',axis=1)  # independent variable
y = df_encoded['deposit']              # dependent variable
print(x.shape)
print(y.shape)
print(type(x))
print(type(y))


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[29]:



from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def eval_model(y_test,y_pred):
    acc = accuracy_score(y_test,y_pred)
    print('Accuracy_Score',acc)
    cm = confusion_matrix(y_test,y_pred)
    print('Confusion Matrix\n',cm)
    print('Classification Report\n',classification_report(y_test,y_pred))

def mscore(model):
    train_score = model.score(x_train,y_train)
    test_score = model.score(x_test,y_test)
    print('Training Score',train_score)  
    print('Testing Score',test_score)


# In[30]:



from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=10)
dt.fit(x_train,y_train)


# In[31]:


mscore(dt)


# In[32]:



ypred_dt = dt.predict(x_test)
print(ypred_dt)


# In[33]:


eval_model(y_test,ypred_dt)


# In[34]:


from sklearn.tree import plot_tree


# In[35]:


cn = ['no','yes']
fn = x_train.columns
print(fn)
print(cn)


# In[36]:


plt.figure(figsize=(30,10))
plot_tree(dt,class_names=cn,filled=True)
plt.show()


# In[37]:


dt1 = DecisionTreeClassifier(criterion='entropy',max_depth=4,min_samples_split=15)
dt1.fit(x_train,y_train)


# In[38]:


mscore(dt1)


# In[39]:


ypred_dt1 = dt1.predict(x_test)


# In[40]:


eval_model(y_test,ypred_dt1)


# In[41]:


plt.figure(figsize=(40,20))
plot_tree(dt1,class_names=cn,filled=True)
plt.show()


# In[ ]:




