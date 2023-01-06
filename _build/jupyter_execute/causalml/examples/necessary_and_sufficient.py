#!/usr/bin/env python
# coding: utf-8

# # Calculating the probabilities of necessary and sufficient causation
# 
# Consider the causal effect of a voucher on customer conversion. We can distinguish between the following types of causation:
# 
# * **Necessary**: If the customer doesn't get the voucher, they will not convert
# * **Sufficient**: If the customer gets the voucher, they will convert
# * **Necessary and sufficient**: The customer will convert if and only if they receive the voucher
# 
# In general, we would like many intervetions to be of the last type. If the voucher is not necessary for a given customer, we might be wasting money by targeting them; if the voucher is not sufficient, we may not fulfil the goal of the campaign, which is to cause customers to convert.
# 
# [Tian and Pearl (2000)](https://ftp.cs.ucla.edu/pub/stat_ser/r271-A.pdf) provided a way to combine experimental and observational data to derive bounds for the probability of each of the above types of causation. In this notebook, we replicate the example from their paper. 

# In[1]:


import numpy as np
import pandas as pd

from causalml.optimize import get_pns_bounds


# [Tian and Pearl (2000, p. 306)](https://ftp.cs.ucla.edu/pub/stat_ser/r271-A.pdf) imagine a setup where we have both experimental and observational data about the efficacy of a certain drug. The experimental data looks as follows:
# 
# |           | Treatment | Control |
# |-----------|-----------|---------|
# | Deaths    | 16        | 14      |
# | Survivals | 984       | 986     |
# 
# Therefore, based on the experiment, it looks like there isn't much of a difference in the rate of deaths in the treatment and control groups. However, in addition to the experimental data, we also have the following data that is from an observational study, i.e. a study in which we simply observe the outcomes for those who choose to use the drug vs. those who don't:
# 
# |           | Treatment | Control |
# |-----------|-----------|---------|
# | Deaths    | 2         | 28      |
# | Survivals | 998       | 972     |
# 
# Because people self-select to use the drug, the data shown in the table is very likely confounded. However, Tian and Pearl argue that the above two datasets can be combined to obtain information that is not visible by looking at either of the datasets independently, namely the probabilities of necessary and sufficient causation (PNS). More specifically, it is possible to derive bounds for PNS by combining the two data sources. To see how, let's generate the datasets:

# In[2]:


num_samples = 2000
half = int(num_samples / 2)
treatment = np.tile([0, 1], half)
recovery = np.zeros(num_samples)

df_rct = pd.DataFrame({'treatment': treatment, 'death': recovery})
df_obs = pd.DataFrame({'treatment': treatment, 'death': recovery})


# In[3]:


# Set the label to `1' for 16 treatment and 14 control observations
df_rct.loc[df_rct.loc[df_rct['treatment'] == 1].sample(n=16).index, 'death'] = 1
df_rct.loc[df_rct.loc[df_rct['treatment'] == 0].sample(n=14).index, 'death'] = 1


# In[4]:


df_rct.groupby('treatment')['death'].sum()


# In[5]:


# Set the label to `1' for 2 treatment and 28 control observations
df_obs.loc[df_obs.loc[df_obs['treatment'] == 1].sample(n=2).index, 'death'] = 1
df_obs.loc[df_obs.loc[df_obs['treatment'] == 0].sample(n=28).index, 'death'] = 1


# In[6]:


df_obs.groupby('treatment')['death'].sum()


# WIth these data, we can now use the `get_pns_bounds()' function to calculate the relevant bounds. Let's do it for each of the three types of bound:

# In[7]:


pns_lb, pns_ub = get_pns_bounds(df_rct, df_obs, 'treatment', 'death', type='PNS')


# In[8]:


pn_lb, pn_ub = get_pns_bounds(df_rct, df_obs, 'treatment', 'death', type='PN')


# In[9]:


ps_lb, ps_ub = get_pns_bounds(df_rct, df_obs, 'treatment', 'death', type='PS')


# In[10]:


print(f'''
Bounds for the probability of necessary causation: [{round(pn_lb, 3)}, {round(pn_ub, 3)}]
Bounds for the probability of sufficient causation: [{round(ps_lb, 3)}, {round(ps_ub, 3)}]
Bounds for the probability of necessary and sufficient causation: [{round(pns_lb, 3)}, {round(pns_ub, 3)}]
''')


# So, by combining experimental and observational data, we arrive at the conclusion that the participants who died and took the drug would have definitely survived without taking the drug. Those who survived and did not take the drug would have had between 0.2% and 3.1% risk of dying had they taken the drug. This illustrates how combining experimental and observational data can lead to additional insights compared to analysing either data source separately.
