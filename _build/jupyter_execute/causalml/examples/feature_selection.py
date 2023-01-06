#!/usr/bin/env python
# coding: utf-8

# # Feature Selection for Uplift Modeling
#   
#     
# This notebook includes two sections:  
# - **Feature selection**: demonstrate how to use Filter methods to select the most important numeric features
# - **Performance evaluation**: evaluate the AUUC performance with top features dataset
#   
# *(Paper reference: [Zhao, Zhenyu, et al. "Feature Selection Methods for Uplift Modeling." arXiv preprint arXiv:2005.03447 (2020).](https://arxiv.org/abs/2005.03447))*

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from causalml.dataset import make_uplift_classification


# #### Import FilterSelect class for Filter methods

# In[3]:


from causalml.feature_selection.filters import FilterSelect


# In[4]:


from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from causalml.metrics import plot_gain, auuc_score


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[6]:


import logging

logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)


# ### Generate dataset
# 
# Generate synthetic data using the built-in function.

# In[7]:


# define parameters for simulation

y_name = 'conversion'
treatment_group_keys = ['control', 'treatment1']
n = 10000
n_classification_features = 50
n_classification_informative = 10
n_classification_repeated = 0
n_uplift_increase_dict = {'treatment1': 8}
n_uplift_decrease_dict = {'treatment1': 4}
delta_uplift_increase_dict = {'treatment1': 0.1}
delta_uplift_decrease_dict = {'treatment1': -0.1}

random_seed = 20200808


# In[8]:


df, X_names = make_uplift_classification(
    treatment_name=treatment_group_keys,
    y_name=y_name,
    n_samples=n,
    n_classification_features=n_classification_features,
    n_classification_informative=n_classification_informative,
    n_classification_repeated=n_classification_repeated,
    n_uplift_increase_dict=n_uplift_increase_dict,
    n_uplift_decrease_dict=n_uplift_decrease_dict,
    delta_uplift_increase_dict = delta_uplift_increase_dict, 
    delta_uplift_decrease_dict = delta_uplift_decrease_dict,
    random_seed=random_seed
)


# In[9]:


df.head()


# In[10]:


# Look at the conversion rate and sample size in each group
df.pivot_table(values='conversion',
               index='treatment_group_key',
               aggfunc=[np.mean, np.size],
               margins=True)


# In[11]:


X_names


# ## Feature selection with Filter methods

# ### method = F (F Filter)

# In[12]:


filter_method = FilterSelect() 


# In[13]:


# F Filter with order 1
method = 'F'
f_imp = filter_method.get_importance(df, X_names, y_name, method, 
                      treatment_group = 'treatment1')
f_imp.head()


# In[14]:


# F Filter with order 2
method = 'F'
f_imp = filter_method.get_importance(df, X_names, y_name, method, 
                      treatment_group = 'treatment1', order=2)
f_imp.head()


# In[15]:


# F Filter with order 3
method = 'F'
f_imp = filter_method.get_importance(df, X_names, y_name, method, 
                      treatment_group = 'treatment1', order=3)
f_imp.head()


# ### method = LR (likelihood ratio test)

# In[16]:


# LR Filter with order 1
method = 'LR'
lr_imp = filter_method.get_importance(df, X_names, y_name, method, 
                      treatment_group = 'treatment1')
lr_imp.head()


# In[17]:


# LR Filter with order 2
method = 'LR'
lr_imp = filter_method.get_importance(df, X_names, y_name, method, 
                      treatment_group = 'treatment1',order=2)
lr_imp.head()


# In[18]:


# LR Filter with order 3
method = 'LR'
lr_imp = filter_method.get_importance(df, X_names, y_name, method, 
                      treatment_group = 'treatment1',order=3)
lr_imp.head()


# ### method = KL (KL divergence)

# In[19]:


method = 'KL'
kl_imp = filter_method.get_importance(df, X_names, y_name, method, 
                      treatment_group = 'treatment1',
                      n_bins=10)
kl_imp.head()


# We found all these 3 filter methods were able to rank most of the **informative** and **uplift increase** features on the top.

# ## Performance evaluation  
# 
# Evaluate the AUUC (Area Under the Uplift Curve) score with several uplift models when using top features dataset 

# In[20]:


# train test split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=111)


# In[21]:


# convert treatment column to 1 (treatment1) and 0 (control)
treatments = np.where((df_test['treatment_group_key']=='treatment1'), 1, 0)
print(treatments[:10])
print(df_test['treatment_group_key'][:10])


# ### Uplift RandomForest Classfier

# In[22]:


uplift_model = UpliftRandomForestClassifier(control_name='control', max_depth=8)


# In[23]:


# using all features
features = X_names 
uplift_model.fit(X = df_train[features].values, 
                 treatment = df_train['treatment_group_key'].values,
                 y = df_train[y_name].values)
y_preds = uplift_model.predict(df_test[features].values)


# ### Select top N features based on KL filter

# In[24]:


top_n = 10
top_10_features = kl_imp['feature'][:top_n]
print(top_10_features)


# In[25]:


top_n = 15
top_15_features = kl_imp['feature'][:top_n]
print(top_15_features)


# In[26]:


top_n = 20
top_20_features = kl_imp['feature'][:top_n]
print(top_20_features)


# #### Train the Uplift model again with top N features

# In[27]:


# using top 10 features
features = top_10_features 

uplift_model.fit(X = df_train[features].values, 
                 treatment = df_train['treatment_group_key'].values,
                 y = df_train[y_name].values)
y_preds_t10 = uplift_model.predict(df_test[features].values)


# In[28]:


# using top 15 features
features = top_15_features 

uplift_model.fit(X = df_train[features].values, 
                 treatment = df_train['treatment_group_key'].values,
                 y = df_train[y_name].values)
y_preds_t15 = uplift_model.predict(df_test[features].values)


# In[29]:


# using top 20 features
features = top_20_features

uplift_model.fit(X = df_train[features].values, 
                 treatment = df_train['treatment_group_key'].values,
                 y = df_train[y_name].values)
y_preds_t20 = uplift_model.predict(df_test[features].values)


# ### Print results for Uplift model

# In[30]:


df_preds = pd.DataFrame([y_preds.ravel(), 
                         y_preds_t10.ravel(),
                         y_preds_t15.ravel(),
                         y_preds_t20.ravel(),
                         treatments,
                         df_test[y_name].ravel()],
                        index=['All', 'Top 10', 'Top 15', 'Top 20', 'is_treated', y_name]).T

plot_gain(df_preds, outcome_col=y_name, treatment_col='is_treated')


# In[31]:


auuc_score(df_preds, outcome_col=y_name, treatment_col='is_treated')


# ### R Learner as base and feed in Random Forest Regressor

# In[32]:


r_rf_learner = BaseRRegressor(
    RandomForestRegressor(
        n_estimators = 100,
        max_depth = 8,
        min_samples_leaf = 100
    ), 
control_name='control') 


# In[33]:


# using all features
features = X_names 
r_rf_learner.fit(X = df_train[features].values, 
                 treatment = df_train['treatment_group_key'].values,
                 y = df_train[y_name].values)
y_preds = r_rf_learner.predict(df_test[features].values)


# In[34]:


# using top 10 features
features = top_10_features 
r_rf_learner.fit(X = df_train[features].values, 
                 treatment = df_train['treatment_group_key'].values,
                 y = df_train[y_name].values)
y_preds_t10 = r_rf_learner.predict(df_test[features].values)


# In[35]:


# using top 15 features
features = top_15_features 
r_rf_learner.fit(X = df_train[features].values, 
                 treatment = df_train['treatment_group_key'].values,
                 y = df_train[y_name].values)
y_preds_t15 = r_rf_learner.predict(df_test[features].values)


# In[36]:


# using top 20 features
features = top_20_features 
r_rf_learner.fit(X = df_train[features].values, 
                 treatment = df_train['treatment_group_key'].values,
                 y = df_train[y_name].values)
y_preds_t20 = r_rf_learner.predict(df_test[features].values)


# ### Print results for R Learner

# In[37]:


df_preds = pd.DataFrame([y_preds.ravel(), 
                         y_preds_t10.ravel(),
                         y_preds_t15.ravel(),
                         y_preds_t20.ravel(),
                         treatments,
                         df_test[y_name].ravel()],
                        index=['All', 'Top 10', 'Top 15', 'Top 20', 'is_treated', y_name]).T

plot_gain(df_preds, outcome_col=y_name, treatment_col='is_treated')


# In[38]:


# print out AUUC score
auuc_score(df_preds, outcome_col=y_name, treatment_col='is_treated')


# (a relatively smaller enhancement on the AUUC is observed in this R Learner case)

# ### S Learner as base and feed in Random Forest Regressor

# In[39]:


slearner_rf = BaseSRegressor(
    RandomForestRegressor(
        n_estimators = 100,
        max_depth = 8,
        min_samples_leaf = 100
    ), 
    control_name='control')


# In[40]:


# using all features
features = X_names 
slearner_rf.fit(X = df_train[features].values, 
                treatment = df_train['treatment_group_key'].values,
                y = df_train[y_name].values)
y_preds = slearner_rf.predict(df_test[features].values)


# In[41]:


# using top 10 features
features = top_10_features 
slearner_rf.fit(X = df_train[features].values, 
                treatment = df_train['treatment_group_key'].values,
                y = df_train[y_name].values)
y_preds_t10 = slearner_rf.predict(df_test[features].values)


# In[42]:


# using top 15 features
features = top_15_features 
slearner_rf.fit(X = df_train[features].values, 
                treatment = df_train['treatment_group_key'].values,
                y = df_train[y_name].values)
y_preds_t15 = slearner_rf.predict(df_test[features].values)


# In[43]:


# using top 20 features
features = top_20_features 
slearner_rf.fit(X = df_train[features].values, 
                treatment = df_train['treatment_group_key'].values,
                y = df_train[y_name].values)
y_preds_t20 = slearner_rf.predict(df_test[features].values)


# ### Print results for S Learner

# In[44]:


df_preds = pd.DataFrame([y_preds.ravel(), 
                         y_preds_t10.ravel(),
                         y_preds_t15.ravel(),
                         y_preds_t20.ravel(),
                         treatments,
                         df_test[y_name].ravel()],
                        index=['All', 'Top 10', 'Top 15', 'Top 20', 'is_treated', y_name]).T

plot_gain(df_preds, outcome_col=y_name, treatment_col='is_treated')


# In[45]:


# print out AUUC score
auuc_score(df_preds, outcome_col=y_name, treatment_col='is_treated')


# In this notebook, we demonstrated how our Filter method functions are able to select important features and enhance the AUUC performance (while the results might vary among different datasets, models and hyper-parameters).

# In[ ]:




