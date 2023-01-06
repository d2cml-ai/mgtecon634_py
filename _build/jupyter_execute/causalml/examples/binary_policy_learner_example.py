#!/usr/bin/env python
# coding: utf-8

# # Policy Learning Notebook

# This notebook demonstrates the use of the CausalML implementation of the policy learner by Athey and Wager (2018) (https://arxiv.org/abs/1702.02896).

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[3]:


from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[4]:


from causalml.optimize import PolicyLearner
from sklearn.tree import plot_tree
from lightgbm import LGBMRegressor
from causalml.inference.meta import BaseXRegressor


# # Binary treatment policy learning

# First we generate a synthetic data set with binary treatment. The treatment is random conditioned on covariates. The treatment effect is heterogeneous where for some individuals it is negative. We use a policy learner to classify the individuals into treat/no-treat groups to maximize the total treatment effect.  

# In[5]:


np.random.seed(1234)

n = 10000
p = 10

X = np.random.normal(size=(n, p))
ee = 1 / (1 + np.exp(X[:, 2]))
tt = 1 / (1 + np.exp(X[:, 0] + X[:, 1])/2) - 0.5
W = np.random.binomial(1, ee, n)
Y = X[:, 2] + W * tt + np.random.normal(size=n)


# Use policy learner with default outcome/treatment estimator and a simple policy classifier.

# In[6]:


policy_learner = PolicyLearner(policy_learner=DecisionTreeClassifier(max_depth=2), calibration=True)


# In[7]:


policy_learner.fit(X, W, Y)


# In[8]:


plt.figure(figsize=(15,7))
plot_tree(policy_learner.model_pi)


# Alternatively, one can construct a policy directly from the ITE estimated from a X-learner.

# In[9]:


learner_x = BaseXRegressor(LGBMRegressor())
ite_x = learner_x.fit_predict(X=X, treatment=W, y=Y)


# In this example policy learner outperforms the ITE-based policy and gets close to the true optimal.

# In[10]:


pd.DataFrame({
    'DR-DT Optimal': [np.mean((policy_learner.predict(X) + 1) * tt / 2)],
    'True Optimal': [np.mean((np.sign(tt) + 1) * tt / 2)],
    'X Learner': [
        np.mean((np.sign(ite_x) + 1) * tt / 2)
    ],
})


# In[ ]:




