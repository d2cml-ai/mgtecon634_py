#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# CausalML is a Python package that provides a suite of uplift modeling and causal inference methods using machine learning algorithms based on recent research. The package currently supports the following methods:
# 
# Tree-based algorithms
# * Uplift tree/random forests on KL divergence, Euclidean Distance, and Chi-Square
# * Uplift tree/random forests on Contextual Treatment Selection
# 
# Meta-learner algorithms
# * S-learner
# * T-learner
# * X-learner
# * R-learner
# 
# In this notebook, we use synthetic data to demonstrate the use of the tree-based algorithms.

# In[1]:


import numpy as np
import pandas as pd

from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.metrics import plot_gain

from sklearn.model_selection import train_test_split


# In[2]:


import causalml
causalml.__version__


# # Generate synthetic dataset
# 
# The CausalML package contains various functions to generate synthetic datasets for uplift modeling. Here we generate a classification dataset using the make_uplift_classification() function.

# In[3]:


df, x_names = make_uplift_classification()


# In[4]:


df.head()


# In[5]:


# Look at the conversion rate and sample size in each group
df.pivot_table(values='conversion',
               index='treatment_group_key',
               aggfunc=[np.mean, np.size],
               margins=True)


# # Run the uplift random forest classifier
# 
# In this section, we first fit the uplift random forest classifier using training data. We then use the fitted model to make a prediction using testing data. The prediction returns an ndarray in which each column contains the predicted uplift if the unit was in the corresponding treatment group.

# In[6]:


# Split data to training and testing samples for model validation (next section)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=111)


# In[7]:


from causalml.inference.tree import UpliftTreeClassifier


# In[8]:


clf = UpliftTreeClassifier(control_name='control')
clf.fit(df_train[x_names].values,
         treatment=df_train['treatment_group_key'].values,
         y=df_train['conversion'].values)
p = clf.predict(df_test[x_names].values)


# In[9]:


df_res = pd.DataFrame(p, columns=clf.classes_)
df_res.head()


# In[10]:


uplift_model = UpliftRandomForestClassifier(control_name='control')


# In[11]:


uplift_model.fit(df_train[x_names].values,
                 treatment=df_train['treatment_group_key'].values,
                 y=df_train['conversion'].values)


# In[12]:


df_res = uplift_model.predict(df_test[x_names].values, full_output=True)
print(df_res.shape)
df_res.head()


# In[13]:


y_pred = uplift_model.predict(df_test[x_names].values)


# In[14]:


y_pred.shape


# In[15]:


# Put the predictions to a DataFrame for a neater presentation
# The output of `predict()` is a numpy array with the shape of [n_sample, n_treatment] excluding the
# predictions for the control group.
result = pd.DataFrame(y_pred,
                      columns=uplift_model.classes_[1:])
result.head()


# # Create the uplift curve
# 
# The performance of the model can be evaluated with the help of the [uplift curve](http://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf). 

# ## Create a synthetic population
# 
# The uplift curve is calculated on a synthetic population that consists of those that were in the control group and those who happened to be in the treatment group recommended by the model. We use the synthetic population to calculate the _actual_ treatment effect within _predicted_ treatment effect quantiles. Because the data is randomized, we have a roughly equal number of treatment and control observations in the predicted quantiles and there is no self selection to treatment groups.

# In[16]:


# If all deltas are negative, assing to control; otherwise assign to the treatment
# with the highest delta
best_treatment = np.where((result < 0).all(axis=1),
                           'control',
                           result.idxmax(axis=1))

# Create indicator variables for whether a unit happened to have the
# recommended treatment or was in the control group
actual_is_best = np.where(df_test['treatment_group_key'] == best_treatment, 1, 0)
actual_is_control = np.where(df_test['treatment_group_key'] == 'control', 1, 0)


# In[17]:


synthetic = (actual_is_best == 1) | (actual_is_control == 1)
synth = result[synthetic]


# ## Calculate the observed treatment effect per predicted treatment effect quantile
# 
# We use the observed treatment effect to calculate the uplift curve, which answers the question: how much of the total cumulative uplift could we have captured by targeting a subset of the population sorted according to the predicted uplift, from highest to lowest?
# 
# CausalML has the plot_gain() function which calculates the uplift curve given a DataFrame containing the treatment assignment, observed outcome and the predicted treatment effect.

# In[18]:


auuc_metrics = (synth.assign(is_treated = 1 - actual_is_control[synthetic],
                             conversion = df_test.loc[synthetic, 'conversion'].values,
                             uplift_tree = synth.max(axis=1))
                     .drop(columns=list(uplift_model.classes_[1:])))


# In[19]:


plot_gain(auuc_metrics, outcome_col='conversion', treatment_col='is_treated')


# In[ ]:




