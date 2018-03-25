
# coding: utf-8

# In[2]:


import os
import pandas as pd
data = pd.read_csv("Customer Churn Data.csv")
data = data.drop(["Id", "state", "phone_number"], axis = 1)
data["international_plan"] = data["international_plan"].map({" yes": True, " no": False})
data["voice_mail_plan"] = data["voice_mail_plan"].map({" yes": True, " no": False})
data["churn"] = data["churn"].map({" True": True, " False": False})

import numpy as np
from sklearn.model_selection import train_test_split

data_true = data[data["churn"]]
data_false = data[~data["churn"]]

train_true, test_true = train_test_split(data_true, test_size=0.2)
train_false, test_false = train_test_split(data_false, test_size=0.2)

train = pd.concat([train_true, train_false])
test = pd.concat([test_true, test_false])

train = np.random.permutation(train)
test = np.random.permutation(test)

train_X = train[:,0:17]
train_y = train[:,18].astype(bool)

test_X = test[:,0:17]
test_y = test[:,18].astype(bool)

total = len(test_X)

len(train_X), len(test_X), len(train_y), len(test_y)

