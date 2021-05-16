#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


from ml_toolbox.nn import NeuralNetwork
from ml_toolbox.lr import LogisticRegression
from ml_toolbox.gnb import NaiveBayes
from ml_toolbox.dtree import DecisionTree
from ml_toolbox.utils.data import read_csv_data


# In[6]:


import numpy as np
from sklearn.neural_network import MLPClassifier


# In[9]:


train_filename = '../data/smallTrain.csv'
valid_filename = '../data/smallValidation.csv'


# In[10]:


## Read data
train_labels, train_features = read_csv_data(train_filename)
valid_labels, valid_features = read_csv_data(valid_filename)


# X = np.array([[0., 0.], [1., 1.]])
# X_valid = np.array([[0., 1.]])
# y = np.array([0, 1])


# In[25]:


nn = NeuralNetwork(train_features, train_labels, learning_rate=0.1, num_classes=10, hidden_units=4, epochs=2, flag= 2)


# In[31]:


get_ipython().run_line_magic('pinfo', 'LogisticRegression')


# In[32]:


lr = LogisticRegression(train_features, train_labels, learning_rate=0.1, num_classes=10, hidden_units=4, epochs=100)


# In[26]:


nn.train(train_features, train_labels, 'metrics.out')
# nn.predict()
# nn.y_preds


# In[29]:


nn.validation_err


# In[52]:


train_labels


# In[70]:


## test against Sckit Learn
clf = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(4,), random_state=1)
clf.fit(train_features, train_labels)
# clf.predict([[2., 2.], [-1., -2.]])


# In[ ]:




