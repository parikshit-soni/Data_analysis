
# coding: utf-8

# In[1]:


import os
import pandas as pd
data = pd.read_csv("Customer Churn Data.csv")
data = data.drop(["Id", "state", "phone_number"], axis = 1)
data["international_plan"] = data["international_plan"].map({" yes": True, " no": False})
data["voice_mail_plan"] = data["voice_mail_plan"].map({" yes": True, " no": False})
data["churn"] = data["churn"].map({" True": True, " False": False})
data

