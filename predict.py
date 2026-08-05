#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import pandas as pd


# In[2]:


model = joblib.load("telecom_tower_model.pkl")


# In[3]:


new_data = pd.DataFrame({
    'Temperature_C':[55],
    'Battery_Voltage':[11.2],
    'Power_Consumption_W':[2543],
    'Signal_Strength_Percent':[55],
    'Fan_Speed_RPM':[2367],
    'Humidity_Percent':[59],
    'Traffic_Load':[4532],
    'Tower_Age_Years':[5]
    })


# In[4]:


prediction = model.predict(new_data)


# In[16]:


result = "Hardware Failure" if prediction[0] == 1 else "Tower Healthy"


# In[17]:


print(result)


# In[ ]:




