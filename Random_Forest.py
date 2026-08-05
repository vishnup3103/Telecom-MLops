#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import joblib


# In[7]:


df = pd.read_csv("Telecom_Tower_Failure_Dataset_10000-1.csv")


# In[8]:


df.head()


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[10]:


import json


# In[13]:


df.columns


# In[ ]:





# In[20]:


X =df[['Temperature_C', 'Battery_Voltage', 'Power_Consumption_W',
       'Signal_Strength_Percent', 'Fan_Speed_RPM', 'Humidity_Percent',
       'Traffic_Load', 'Tower_Age_Years']]
y= df['Failure_Within_48Hrs']


# In[21]:


X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[26]:


model = RandomForestClassifier(n_estimators=50,random_state=42)


# In[27]:


model.fit(X_train,y_train)


# In[28]:


accuracy = model.score(X_test,y_test)


# In[29]:


print(accuracy)


# In[30]:


joblib.dump(model,"telecom_tower_model.pkl")


# In[31]:


metrics = {"accuracy":accuracy}


# In[33]:


with open("metrics.json","w") as f:
    json.dump(metrics,f,indent=4)
print("Training completed Successfully")


print ("Hello World")
