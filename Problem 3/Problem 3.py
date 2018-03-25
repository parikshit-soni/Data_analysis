
# coding: utf-8

# In[9]:


import os
import pandas as pd
data = pd.read_csv("Customer Churn Data.csv")
data = data.drop(["Id", "state", "phone_number"], axis = 1)
data["international_plan"] = data["international_plan"].map({" yes": True, " no": False})
data["voice_mail_plan"] = data["voice_mail_plan"].map({" yes": True, " no": False})
data["churn"] = data["churn"].map({" True": True, " False": False})
data

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



# In[10]:



from sklearn.naive_bayes import GaussianNB

Gaussian_NB = GaussianNB()
Gaussian_NB.fit(train_X, train_y)
Gaussian_NB_pred = Gaussian_NB.predict(test_X)
Gaussian_NB_correct = np.count_nonzero(Gaussian_NB_pred == test_y)

print "Accuracy of Naive Bayes: ", Gaussian_NB_correct*100.0/total, "%"



# In[11]:


from sklearn import svm

svm_classifier = svm.SVC(kernel = "rbf")
svm_classifier.fit(train_X, train_y)
svm_pred = svm_classifier.predict(test_X)
svm_correct = np.count_nonzero(svm_pred == test_y)

print "Accuracy of Linear SVM: ", svm_correct*100.0/total, "%"


# In[12]:


from sklearn import tree

tree_classifier = tree.DecisionTreeClassifier()
tree_classifier = tree_classifier.fit(train_X, train_y)
pred = tree_classifier.predict(test_X)
correct = np.count_nonzero(pred == test_y)

print "Accuracy of Decision Tree: ", correct*100.0/total, "%"

