#!/usr/bin/env python
# coding: utf-8

# # 2SLS Estimation Examples

# We demonstrate the use of 2SLS from the package to estimate the average treatment effect by semi-synthetic data and full synthetic data.

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os
base_path = os.path.abspath("../")
os.chdir(base_path)


# In[52]:


import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
from scipy import stats


# In[39]:


import causalml
from causalml.inference.iv import IVRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


# ## Semi-Synthetic Data from NLSYM

# The data generation mechanism is described in Syrgkanis et al "*Machine Learning Estimation of Heterogeneous Treatment Effects with Instruments*" (2019).

# ### Data Loading

# In[5]:


df = pd.read_csv("examples/data/card.csv")


# In[6]:


df.head()


# In[7]:


df.columns.values


# In[30]:


data_filter = df['educ'] >= 6
# outcome variable
y=df[data_filter]['lwage'].values
# treatment variable
treatment=df[data_filter]['educ'].values
# instrumental variable
w=df[data_filter]['nearc4'].values

Xdf=df[data_filter][['fatheduc', 'motheduc', 'momdad14', 'sinmom14', 'reg661', 'reg662',
      'reg663', 'reg664', 'reg665', 'reg666', 'reg667', 'reg668',
      'reg669', 'south66', 'black', 'smsa', 'south', 'smsa66',
      'exper', 'expersq']]
Xdf['fatheduc']=Xdf['fatheduc'].fillna(value=Xdf['fatheduc'].mean())
Xdf['motheduc']=Xdf['motheduc'].fillna(value=Xdf['motheduc'].mean())
Xscale=Xdf.copy()
Xscale[['fatheduc', 'motheduc', 'exper', 'expersq']]=StandardScaler().fit_transform(Xscale[['fatheduc', 'motheduc', 'exper', 'expersq']])
X=Xscale.values


# In[32]:


Xscale.describe()


# ### Semi-Synthetic Data Generation

# In[29]:


def semi_synth_nlsym(X, w, random_seed=None):
    np.random.seed(random_seed)
    nobs = X.shape[0]
    nv = np.random.uniform(0, 1, size=nobs)
    c0 = np.random.uniform(0.2, 0.3)
    C = c0 * X[:,1]
    # Treatment compliance depends on mother education
    treatment = C * w + X[:,1] + nv
    # Treatment effect depends no mother education and single-mom family at age 14
    theta = 0.1 + 0.05 * X[:,1] - 0.1*X[:,3]
    # Additional effect on the outcome from mother education
    f = 0.05 * X[:,1]
    y = theta * (treatment + nv) + f + np.random.normal(0, 0.1, size=nobs)
    
    return y, treatment, theta


# In[33]:


y_sim, treatment_sim, theta = semi_synth_nlsym(Xdf.values, w)


# ### Estimation

# In[36]:


# True value
theta.mean()


# In[38]:


# 2SLS estimate
iv_fit = IVRegressor()
iv_fit.fit(X, treatment_sim, y_sim, w)
ate, ate_sd = iv_fit.predict()
(ate, ate_sd)


# In[51]:


# OLS estimate
ols_fit=sm.OLS(y_sim, sm.add_constant(np.c_[treatment_sim, X], prepend=False)).fit()
(ols_fit.params[0], ols_fit.bse[0])


# ## Pure Synthetic Data

# The data generation mechanism is described in Hong et al "*Semiparametric Efficiency in Nonlinear LATE Models*" (2010).

# ### Data Generation

# In[54]:


def synthetic_data(n=10000, random_seed=None):
    np.random.seed(random_seed)
    gamma0 = -0.5
    gamma1 = 1.0
    delta = 1.0
    x = np.random.uniform(size=n)
    v = np.random.normal(size=n)
    d1 = (gamma0 + x*gamma1 + delta + v>=0).astype(float)
    d0 = (gamma0 + x*gamma1 + v>=0).astype(float)
    
    alpha = 1.0
    beta = 0.5
    lambda11 = 2.0
    lambda00 = 1.0
    xi1 = np.random.poisson(np.exp(alpha+x*beta))
    xi2 = np.random.poisson(np.exp(x*beta))
    xi3 = np.random.poisson(np.exp(lambda11), size=n)
    xi4 = np.random.poisson(np.exp(lambda00), size=n)
    
    y1 = xi1 + xi3 * ((d1==1) & (d0==1)) + xi4 * ((d1==0) & (d0==0))
    y0 = xi2 + xi3 * ((d1==1) & (d0==1)) + xi4 * ((d1==0) & (d0==0))
    
    z = np.random.binomial(1, stats.norm.cdf(x))
    d = d1*z + d0*(1-z)
    y = y1*d + y0*(1-d)
    
    return y, x, d, z, y1[(d1>d0)].mean()-y0[(d1>d0)].mean()


# In[55]:


y, x, d, z, late = synthetic_data()


# ### Estimation

# In[56]:


# True value
late


# In[57]:


# 2SLS estimate
iv_fit = IVRegressor()
iv_fit.fit(x, d, y, z)
ate, ate_sd = iv_fit.predict()
(ate, ate_sd)


# In[59]:


# OLS estimate
ols_fit=sm.OLS(y, sm.add_constant(np.c_[d, x], prepend=False)).fit()
(ols_fit.params[0], ols_fit.bse[0])


# In[ ]:




