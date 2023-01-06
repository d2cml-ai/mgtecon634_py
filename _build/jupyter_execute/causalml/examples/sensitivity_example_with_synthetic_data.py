#!/usr/bin/env python
# coding: utf-8

# # Setup
# In this notebook, we will generate some synthetic data to demonstrate how sensitivity analysis work by different methods.
# 
# # Sensitivity Analysis
# ## Methods
# We provided five methods for sensitivity analysis including (Placebb Treatment, Random Cause, Subset Data, Random Replace and Selection Bias). 
# This notebook will walkthrough how to use the combined function sensitivity_analysis() to compare different method and also how to use each individual method separately:
# 
# 1. Placebo Treatment: Replacing treatment with a random variable
# 2. Irrelevant Additional Confounder: Adding a random common cause variable
# 3. Subset validation: Removing a random subset of the data
# 4. Selection Bias method with One Sided confounding function and Alignment confounding function
# 5. Random Replace: Random replace a covariate with an irrelevant variable

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib
from causalml.inference.meta import BaseXLearner
from causalml.dataset import synthetic_data

from causalml.metrics.sensitivity import Sensitivity
from causalml.metrics.sensitivity import SensitivityRandomReplace, SensitivitySelectionBias

plt.style.use('fivethirtyeight')
matplotlib.rcParams['figure.figsize'] = [8, 8]
warnings.filterwarnings('ignore')

# logging.basicConfig(level=logging.INFO)

pd.options.display.float_format = '{:.4f}'.format


# # Data Pred

# ## Generate Synthetic data

# In[4]:


# Generate synthetic data using mode 1
num_features = 6 
y, X, treatment, tau, b, e = synthetic_data(mode=1, n=100000, p=num_features, sigma=1.0)


# In[5]:


tau.mean()


# ## Define Features

# In[6]:


# Generate features names
INFERENCE_FEATURES = ['feature_' + str(i) for i in range(num_features)]
TREATMENT_COL = 'target'
OUTCOME_COL = 'outcome'
SCORE_COL = 'pihat'


# In[7]:


df = pd.DataFrame(X, columns=INFERENCE_FEATURES)
df[TREATMENT_COL] = treatment
df[OUTCOME_COL] = y
df[SCORE_COL] = e


# In[8]:


df.head()


# # Sensitivity Analysis

# ## With all Covariates

# ### Sensitivity Analysis Summary Report (with One-sided confounding function and default alpha)

# In[9]:


# Calling the Base XLearner class and return the sensitivity analysis summary report
learner_x = BaseXLearner(LinearRegression())
sens_x = Sensitivity(df=df, inference_features=INFERENCE_FEATURES, p_col='pihat',
                     treatment_col=TREATMENT_COL, outcome_col=OUTCOME_COL, learner=learner_x)
# Here for Selection Bias method will use default one-sided confounding function and alpha (quantile range of outcome values) input
sens_sumary_x = sens_x.sensitivity_analysis(methods=['Placebo Treatment',
                                                     'Random Cause',
                                                     'Subset Data',
                                                     'Random Replace',
                                                     'Selection Bias'], sample_size=0.5)


# In[10]:


# From the following results, refutation methods show our model is pretty robust; 
# When alpah > 0, the treated group always has higher mean potential outcomes than the control; when  < 0, the control group is better off.
sens_sumary_x


# ### Random Replace

# In[11]:


# Replace feature_0 with an irrelevent variable
sens_x_replace = SensitivityRandomReplace(df=df, inference_features=INFERENCE_FEATURES, p_col='pihat',
                                          treatment_col=TREATMENT_COL, outcome_col=OUTCOME_COL, learner=learner_x,
                                          sample_size=0.9, replaced_feature='feature_0')
s_check_replace = sens_x_replace.summary(method='Random Replace')
s_check_replace


# ### Selection Bias: Alignment confounding Function

# In[12]:


sens_x_bias_alignment = SensitivitySelectionBias(df, INFERENCE_FEATURES, p_col='pihat', treatment_col=TREATMENT_COL,
                                                 outcome_col=OUTCOME_COL, learner=learner_x, confound='alignment',
                                                 alpha_range=None)


# In[13]:


lls_x_bias_alignment, partial_rsqs_x_bias_alignment = sens_x_bias_alignment.causalsens()


# In[14]:


lls_x_bias_alignment


# In[15]:


partial_rsqs_x_bias_alignment


# In[16]:


# Plot the results by confounding vector and plot Confidence Intervals for ATE
sens_x_bias_alignment.plot(lls_x_bias_alignment, ci=True)


# In[17]:


# Plot the results by rsquare with partial r-square results by each individual features
sens_x_bias_alignment.plot(lls_x_bias_alignment, partial_rsqs_x_bias_alignment, type='r.squared', partial_rsqs=True)


# ## Drop One Confounder

# In[18]:


df_new = df.drop('feature_0', axis=1).copy()
INFERENCE_FEATURES_new = INFERENCE_FEATURES.copy()
INFERENCE_FEATURES_new.remove('feature_0')
df_new.head()


# In[19]:


INFERENCE_FEATURES_new


# ### Sensitivity Analysis Summary Report (with One-sided confounding function and default alpha)

# In[20]:


sens_x_new = Sensitivity(df=df_new, inference_features=INFERENCE_FEATURES_new, p_col='pihat',
                     treatment_col=TREATMENT_COL, outcome_col=OUTCOME_COL, learner=learner_x)
# Here for Selection Bias method will use default one-sided confounding function and alpha (quantile range of outcome values) input
sens_sumary_x_new = sens_x_new.sensitivity_analysis(methods=['Placebo Treatment',
                                                     'Random Cause',
                                                     'Subset Data',
                                                     'Random Replace',
                                                     'Selection Bias'], sample_size=0.5)


# In[21]:


# Here we can see the New ATE restul from Random Replace method actually changed ~ 12.5%
sens_sumary_x_new


# ### Random Replace

# In[22]:


# Replace feature_0 with an irrelevent variable
sens_x_replace_new = SensitivityRandomReplace(df=df_new, inference_features=INFERENCE_FEATURES_new, p_col='pihat',
                                          treatment_col=TREATMENT_COL, outcome_col=OUTCOME_COL, learner=learner_x,
                                          sample_size=0.9, replaced_feature='feature_1')
s_check_replace_new = sens_x_replace_new.summary(method='Random Replace')
s_check_replace_new


# ### Selection Bias: Alignment confounding Function

# In[23]:


sens_x_bias_alignment_new = SensitivitySelectionBias(df_new, INFERENCE_FEATURES_new, p_col='pihat', treatment_col=TREATMENT_COL,
                                                 outcome_col=OUTCOME_COL, learner=learner_x, confound='alignment',
                                                 alpha_range=None)


# In[24]:


lls_x_bias_alignment_new, partial_rsqs_x_bias_alignment_new = sens_x_bias_alignment_new.causalsens()


# In[25]:


lls_x_bias_alignment_new


# In[26]:


partial_rsqs_x_bias_alignment_new


# In[27]:


# Plot the results by confounding vector and plot Confidence Intervals for ATE
sens_x_bias_alignment_new.plot(lls_x_bias_alignment_new, ci=True)


# In[28]:


# Plot the results by rsquare with partial r-square results by each individual features
sens_x_bias_alignment_new.plot(lls_x_bias_alignment_new, partial_rsqs_x_bias_alignment_new, type='r.squared', partial_rsqs=True)


# ## Generate a Selection Bias Set

# In[29]:


df_new_2 = df.copy()
df_new_2['treated_new'] = df['feature_0'].rank()
df_new_2['treated_new'] = [1 if i > df_new_2.shape[0]/2 else 0 for i in df_new_2['treated_new']]


# In[30]:


df_new_2.head()


# ### Sensitivity Analysis Summary Report (with One-sided confounding function and default alpha)

# In[31]:


sens_x_new_2 = Sensitivity(df=df_new_2, inference_features=INFERENCE_FEATURES, p_col='pihat',
                     treatment_col='treated_new', outcome_col=OUTCOME_COL, learner=learner_x)
# Here for Selection Bias method will use default one-sided confounding function and alpha (quantile range of outcome values) input
sens_sumary_x_new_2 = sens_x_new_2.sensitivity_analysis(methods=['Placebo Treatment',
                                                     'Random Cause',
                                                     'Subset Data',
                                                     'Random Replace',
                                                     'Selection Bias'], sample_size=0.5)


# In[32]:


sens_sumary_x_new_2


# ### Random Replace

# In[33]:


# Replace feature_0 with an irrelevent variable
sens_x_replace_new_2 = SensitivityRandomReplace(df=df_new_2, inference_features=INFERENCE_FEATURES, p_col='pihat',
                                          treatment_col='treated_new', outcome_col=OUTCOME_COL, learner=learner_x,
                                          sample_size=0.9, replaced_feature='feature_0')
s_check_replace_new_2 = sens_x_replace_new_2.summary(method='Random Replace')
s_check_replace_new_2


# ### Selection Bias: Alignment confounding Function

# In[34]:


sens_x_bias_alignment_new_2 = SensitivitySelectionBias(df_new_2, INFERENCE_FEATURES, p_col='pihat', treatment_col='treated_new',
                                                 outcome_col=OUTCOME_COL, learner=learner_x, confound='alignment',
                                                 alpha_range=None)


# In[35]:


lls_x_bias_alignment_new_2, partial_rsqs_x_bias_alignment_new_2 = sens_x_bias_alignment_new_2.causalsens()


# In[36]:


lls_x_bias_alignment_new_2


# In[37]:


partial_rsqs_x_bias_alignment_new_2


# In[38]:


# Plot the results by confounding vector and plot Confidence Intervals for ATE
sens_x_bias_alignment_new_2.plot(lls_x_bias_alignment_new_2, ci=True)


# In[39]:


# Plot the results by rsquare with partial r-square results by each individual features
sens_x_bias_alignment_new_2.plot(lls_x_bias_alignment_new, partial_rsqs_x_bias_alignment_new_2, type='r.squared', partial_rsqs=True)

