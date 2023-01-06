#!/usr/bin/env python
# coding: utf-8

# # `causalml` - Meta-Learner Example Notebook
# This notebook only contains regression examples.

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost import XGBRegressor, XGBClassifier
import warnings

# from causalml.inference.meta import XGBTLearner, MLPTLearner
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier, BaseRClassifier
from causalml.inference.meta import LRSRegressor
from causalml.match import NearestNeighborMatch, MatchOptimizer, create_table_one
from causalml.propensity import ElasticNetPropensityModel
from causalml.dataset import *
from causalml.metrics import *

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# imports from package
import logging
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import statsmodels.api as sm
from copy import deepcopy

logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)

get_ipython().run_line_magic('matplotlib', 'inline')


# # Single Treatment Case

# ### Generate synthetic data

# In[3]:


# Generate synthetic data using mode 1
y, X, treatment, tau, b, e = synthetic_data(mode=1, n=10000, p=8, sigma=1.0)

treatment = np.array(['treatment_a' if val==1 else 'control' for val in treatment])


# ## S-Learner

# ### ATE

# In[4]:


learner_s = BaseSRegressor(XGBRegressor(), control_name='control')
ate_s = learner_s.estimate_ate(X=X, treatment=treatment, y=y, return_ci=False, bootstrap_ci=False)


# In[5]:


ate_s


# ### ATE w/ Confidence Intervals

# In[6]:


alpha = 0.05
learner_s = BaseSRegressor(XGBRegressor(), ate_alpha=alpha, control_name='control')
ate_s, ate_s_lb, ate_s_ub = learner_s.estimate_ate(X=X, treatment=treatment, y=y, return_ci=True,
                                                   bootstrap_ci=False)


# In[7]:


np.vstack((ate_s_lb, ate_s, ate_s_ub))


# ### ATE w/ Boostrap Confidence Intervals

# In[8]:


ate_s_b, ate_s_lb_b, ate_s_ub_b = learner_s.estimate_ate(X=X, treatment=treatment, y=y, return_ci=True,
                                                         bootstrap_ci=True, n_bootstraps=100, bootstrap_size=5000)


# In[9]:


np.vstack((ate_s_lb_b, ate_s_b, ate_s_ub_b))


# ### CATE

# In[10]:


learner_s = BaseSRegressor(XGBRegressor(), control_name='control')
cate_s = learner_s.fit_predict(X=X, treatment=treatment, y=y, return_ci=False)


# In[11]:


cate_s


# ### CATE w/ Confidence Intervals

# In[12]:


alpha = 0.05
learner_s = BaseSRegressor(XGBRegressor(), ate_alpha=alpha, control_name='control')
cate_s, cate_s_lb, cate_s_ub = learner_s.fit_predict(X=X, treatment=treatment, y=y, return_ci=True,
                               n_bootstraps=100, bootstrap_size=5000)


# In[13]:


cate_s


# In[14]:


cate_s_lb


# In[15]:


cate_s_ub


# ## T-Learner

# ### ATE w/ Confidence Intervals

# In[16]:


learner_t = BaseTRegressor(XGBRegressor(), control_name='control')
ate_t, ate_t_lb, ate_t_ub = learner_t.estimate_ate(X=X, treatment=treatment, y=y)


# In[17]:


np.vstack((ate_t_lb, ate_t, ate_t_ub))


# ### ATE w/ Boostrap Confidence Intervals

# In[18]:


ate_t_b, ate_t_lb_b, ate_t_ub_b = learner_t.estimate_ate(X=X, treatment=treatment, y=y, bootstrap_ci=True,
                                                   n_bootstraps=100, bootstrap_size=5000)


# In[19]:


np.vstack((ate_t_lb_b, ate_t_b, ate_t_ub_b))


# ### CATE

# In[20]:


learner_t = BaseTRegressor(XGBRegressor(), control_name='control')
cate_t = learner_t.fit_predict(X=X, treatment=treatment, y=y)


# In[21]:


cate_t


# ### CATE w/ Confidence Intervals

# In[22]:


learner_t = BaseTRegressor(XGBRegressor(), control_name='control')
cate_t, cate_t_lb, cate_t_ub = learner_t.fit_predict(X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=100,
                                                    bootstrap_size=5000)


# In[23]:


cate_t


# In[24]:


cate_t_lb


# In[25]:


cate_t_ub


# ## X-Learner

# ### ATE w/ Confidence Intervals

# #### With Propensity Score Input

# In[26]:


learner_x = BaseXRegressor(XGBRegressor(), control_name='control')
ate_x, ate_x_lb, ate_x_ub = learner_x.estimate_ate(X=X, treatment=treatment, y=y, p=e)


# In[27]:


np.vstack((ate_x_lb, ate_x, ate_x_ub))


# #### Without Propensity Score input

# In[28]:


ate_x_no_p, ate_x_lb_no_p, ate_x_ub_no_p = learner_x.estimate_ate(X=X, treatment=treatment, y=y)


# In[29]:


np.vstack((ate_x_lb_no_p, ate_x_no_p, ate_x_ub_no_p))


# In[30]:


learner_x.propensity_model


# ### ATE w/ Boostrap Confidence Intervals

# #### With Propensity Score Input

# In[31]:


ate_x_b, ate_x_lb_b, ate_x_ub_b = learner_x.estimate_ate(X=X, treatment=treatment, y=y, p=e, bootstrap_ci=True,
                                                   n_bootstraps=100, bootstrap_size=5000)


# In[32]:


np.vstack((ate_x_lb_b, ate_x_b, ate_x_ub_b))


# #### Without Propensity Score Input

# In[33]:


ate_x_b_no_p, ate_x_lb_b_no_p, ate_x_ub_b_no_p = learner_x.estimate_ate(X=X, treatment=treatment, y=y, bootstrap_ci=True,
                                                   n_bootstraps=100, bootstrap_size=5000)


# In[34]:


np.vstack((ate_x_lb_b_no_p, ate_x_b_no_p, ate_x_ub_b_no_p))


# ### CATE

# #### With Propensity Score Input

# In[35]:


learner_x = BaseXRegressor(XGBRegressor(), control_name='control')
cate_x = learner_x.fit_predict(X=X, treatment=treatment, y=y, p=e)


# In[36]:


cate_x


# #### Without Propensity Score Input

# In[37]:


cate_x_no_p = learner_x.fit_predict(X=X, treatment=treatment, y=y)


# In[38]:


cate_x_no_p


# ### CATE w/ Confidence Intervals

# #### With Propensity Score Input

# In[39]:


learner_x = BaseXRegressor(XGBRegressor(), control_name='control')
cate_x, cate_x_lb, cate_x_ub = learner_x.fit_predict(X=X, treatment=treatment, y=y, p=e, return_ci=True,
                                                     n_bootstraps=100, bootstrap_size=3000)


# In[40]:


cate_x


# In[41]:


cate_x_lb


# In[42]:


cate_x_ub


# #### Without Propensity Score Input

# In[43]:


cate_x_no_p, cate_x_lb_no_p, cate_x_ub_no_p = learner_x.fit_predict(X=X, treatment=treatment, y=y, return_ci=True,
                                                     n_bootstraps=100, bootstrap_size=3000)


# In[44]:


cate_x_no_p


# In[45]:


cate_x_lb_no_p


# In[46]:


cate_x_ub_no_p


# ## R-Learner

# ### ATE w/ Confidence Intervals

# #### With Propensity Score Input

# In[47]:


learner_r = BaseRRegressor(XGBRegressor(), control_name='control')
ate_r, ate_r_lb, ate_r_ub = learner_r.estimate_ate(X=X, treatment=treatment, y=y, p=e)


# In[48]:


np.vstack((ate_r_lb, ate_r, ate_r_ub))


# #### Without Propensity Score Input

# In[49]:


ate_r_no_p, ate_r_lb_no_p, ate_r_ub_no_p = learner_r.estimate_ate(X=X, treatment=treatment, y=y)


# In[50]:


np.vstack((ate_r_lb_no_p, ate_r_no_p, ate_r_ub_no_p))


# In[51]:


learner_r.propensity_model


# ### ATE w/ Boostrap Confidence Intervals

# #### With Propensity Score Input

# In[52]:


ate_r_b, ate_r_lb_b, ate_r_ub_b = learner_r.estimate_ate(X=X, treatment=treatment, y=y, p=e, bootstrap_ci=True,
                                                   n_bootstraps=100, bootstrap_size=5000)


# In[53]:


np.vstack((ate_r_lb_b, ate_r_b, ate_r_ub_b))


# #### Without Propensity Score Input

# In[54]:


ate_r_b_no_p, ate_r_lb_b_no_p, ate_r_ub_b_no_p = learner_r.estimate_ate(X=X, treatment=treatment, y=y, bootstrap_ci=True,
                                                   n_bootstraps=100, bootstrap_size=5000)


# In[55]:


np.vstack((ate_r_lb_b_no_p, ate_r_b_no_p, ate_r_ub_b_no_p))


# ### CATE

# #### With Propensity Score Input

# In[56]:


learner_r = BaseRRegressor(XGBRegressor(), control_name='control')
cate_r = learner_r.fit_predict(X=X, treatment=treatment, y=y, p=e)


# In[57]:


cate_r


# #### Without Propensity Score Input

# In[58]:


cate_r_no_p = learner_r.fit_predict(X=X, treatment=treatment, y=y)


# In[59]:


cate_r_no_p


# ### CATE w/ Confidence Intervals

# #### With Propensity Score Input

# In[60]:


learner_r = BaseRRegressor(XGBRegressor(), control_name='control')
cate_r, cate_r_lb, cate_r_ub = learner_r.fit_predict(X=X, treatment=treatment, y=y, p=e, return_ci=True,
                                                     n_bootstraps=100, bootstrap_size=1000)


# In[61]:


cate_r


# In[62]:


cate_r_lb


# In[63]:


cate_r_ub


# #### Without Propensity Score Input

# In[64]:


learner_r = BaseRRegressor(XGBRegressor(), control_name='control')
cate_r_no_p, cate_r_lb_no_p, cate_r_ub_no_p = learner_r.fit_predict(X=X, treatment=treatment, y=y, return_ci=True,
                                                     n_bootstraps=100, bootstrap_size=1000)


# In[65]:


cate_r_no_p


# In[66]:


cate_r_lb_no_p


# In[67]:


cate_r_ub_no_p


# # Visualize

# In[68]:


groups = learner_r._classes

alpha = 1
linewidth = 2
bins = 30
for group,idx in sorted(groups.items(), key=lambda x: x[1]):
    plt.figure(figsize=(12,8))
    plt.hist(cate_t[:,idx], alpha=alpha, bins=bins, label='T Learner ({})'.format(group),
             histtype='step', linewidth=linewidth, density=True)
    plt.hist(cate_x[:,idx], alpha=alpha, bins=bins, label='X Learner ({})'.format(group),
             histtype='step', linewidth=linewidth, density=True)
    plt.hist(cate_r[:,idx], alpha=alpha, bins=bins, label='R Learner ({})'.format(group),
             histtype='step', linewidth=linewidth, density=True)
    plt.hist(tau, alpha=alpha, bins=bins, label='Actual ATE distr',
             histtype='step', linewidth=linewidth, color='green', density=True)
    plt.vlines(cate_s[0,idx], 0, plt.axes().get_ylim()[1], label='S Learner ({})'.format(group),
               linestyles='dotted', linewidth=linewidth)
    plt.vlines(tau.mean(), 0, plt.axes().get_ylim()[1], label='Actual ATE',
               linestyles='dotted', linewidth=linewidth, color='green')
    
    plt.title('Distribution of CATE Predictions for {}'.format(group))
    plt.xlabel('Individual Treatment Effect (ITE/CATE)')
    plt.ylabel('# of Samples')
    _=plt.legend()


# ---
# # Multiple Treatment Case

# ### Generate synthetic data
# Note: we randomize the assignment of treatment flag AFTER the synthetic data generation process, so it doesn't make sense to measure accuracy metrics here. Next steps would be to include multi-treatment in the DGP itself.

# In[69]:


# Generate synthetic data using mode 1
y, X, treatment, tau, b, e = synthetic_data(mode=1, n=10000, p=8, sigma=1.0)

treatment = np.array([('treatment_a' if np.random.random() > 0.2 else 'treatment_b') 
                      if val==1 else 'control' for val in treatment])

e = {group: e for group in np.unique(treatment)}


# In[70]:


pd.Series(treatment).value_counts()


# ## S-Learner

# ### ATE

# In[71]:


learner_s = BaseSRegressor(XGBRegressor(), control_name='control')
ate_s = learner_s.estimate_ate(X=X, treatment=treatment, y=y, return_ci=False, bootstrap_ci=False)


# In[72]:


ate_s


# In[73]:


learner_s._classes


# ### ATE w/ Confidence Intervals

# In[74]:


alpha = 0.05
learner_s = BaseSRegressor(XGBRegressor(), ate_alpha=alpha, control_name='control')
ate_s, ate_s_lb, ate_s_ub = learner_s.estimate_ate(X=X, treatment=treatment, y=y, return_ci=True,
                                                   bootstrap_ci=False)


# In[75]:


np.vstack((ate_s_lb, ate_s, ate_s_ub))


# ### ATE w/ Boostrap Confidence Intervals

# In[76]:


ate_s_b, ate_s_lb_b, ate_s_ub_b = learner_s.estimate_ate(X=X, treatment=treatment, y=y, return_ci=True,
                                                         bootstrap_ci=True, n_bootstraps=100, bootstrap_size=5000)


# In[77]:


np.vstack((ate_s_lb_b, ate_s_b, ate_s_ub_b))


# ### CATE

# In[78]:


learner_s = BaseSRegressor(XGBRegressor(), control_name='control')
cate_s = learner_s.fit_predict(X=X, treatment=treatment, y=y, return_ci=False)


# In[79]:


cate_s


# ### CATE w/ Confidence Intervals

# In[80]:


alpha = 0.05
learner_s = BaseSRegressor(XGBRegressor(), ate_alpha=alpha, control_name='control')
cate_s, cate_s_lb, cate_s_ub = learner_s.fit_predict(X=X, treatment=treatment, y=y, return_ci=True,
                               n_bootstraps=100, bootstrap_size=3000)


# In[81]:


cate_s


# In[82]:


cate_s_lb


# In[83]:


cate_s_ub


# ## T-Learner

# ### ATE w/ Confidence Intervals

# In[84]:


learner_t = BaseTRegressor(XGBRegressor(), control_name='control')
ate_t, ate_t_lb, ate_t_ub = learner_t.estimate_ate(X=X, treatment=treatment, y=y)


# In[85]:


np.vstack((ate_t_lb, ate_t, ate_t_ub))


# ### ATE w/ Boostrap Confidence Intervals

# In[86]:


ate_t_b, ate_t_lb_b, ate_t_ub_b = learner_t.estimate_ate(X=X, treatment=treatment, y=y, bootstrap_ci=True,
                                                   n_bootstraps=100, bootstrap_size=5000)


# In[87]:


np.vstack((ate_t_lb_b, ate_t_b, ate_t_ub_b))


# ### CATE

# In[88]:


learner_t = BaseTRegressor(XGBRegressor(), control_name='control')
cate_t = learner_t.fit_predict(X=X, treatment=treatment, y=y)


# In[89]:


cate_t


# ### CATE w/ Confidence Intervals

# In[90]:


learner_t = BaseTRegressor(XGBRegressor(), control_name='control')
cate_t, cate_t_lb, cate_t_ub = learner_t.fit_predict(X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=100,
                                                    bootstrap_size=3000)


# In[91]:


cate_t


# In[92]:


cate_t_lb


# In[93]:


cate_t_ub


# ## X-Learner

# ### ATE w/ Confidence Intervals

# #### With Propensity Score Input

# In[94]:


learner_x = BaseXRegressor(XGBRegressor(), control_name='control')
ate_x, ate_x_lb, ate_x_ub = learner_x.estimate_ate(X=X, treatment=treatment, y=y, p=e)


# In[95]:


np.vstack((ate_x_lb, ate_x, ate_x_ub))


# #### Without Propensity Score Input

# In[96]:


ate_x_no_p, ate_x_lb_no_p, ate_x_ub_no_p = learner_x.estimate_ate(X=X, treatment=treatment, y=y)


# In[97]:


np.vstack((ate_x_lb_no_p, ate_x_no_p, ate_x_ub_no_p))


# ### ATE w/ Boostrap Confidence Intervals

# #### With Propensity Score Input

# In[98]:


ate_x_b, ate_x_lb_b, ate_x_ub_b = learner_x.estimate_ate(X=X, treatment=treatment, y=y, p=e, bootstrap_ci=True,
                                                   n_bootstraps=100, bootstrap_size=5000)


# In[99]:


np.vstack((ate_x_lb_b, ate_x_b, ate_x_ub_b))


# #### Without Propensity Score Input

# In[100]:


ate_x_b_no_p, ate_x_lb_b_no_p, ate_x_ub_b_no_p = learner_x.estimate_ate(X=X, treatment=treatment, y=y, bootstrap_ci=True,
                                                   n_bootstraps=100, bootstrap_size=5000)


# In[101]:


np.vstack((ate_x_lb_b_no_p, ate_x_b_no_p, ate_x_ub_b_no_p))


# ### CATE

# #### With Propensity Score Input

# In[102]:


learner_x = BaseXRegressor(XGBRegressor(), control_name='control')
cate_x = learner_x.fit_predict(X=X, treatment=treatment, y=y, p=e)


# In[103]:


cate_x


# #### Without Propensity Score Input

# In[104]:


cate_x_no_p = learner_x.fit_predict(X=X, treatment=treatment, y=y)


# In[105]:


cate_x_no_p


# ### CATE w/ Confidence Intervals

# #### With Propensity Score Input

# In[106]:


learner_x = BaseXRegressor(XGBRegressor(), control_name='control')
cate_x, cate_x_lb, cate_x_ub = learner_x.fit_predict(X=X, treatment=treatment, y=y, p=e, return_ci=True,
                                                     n_bootstraps=100, bootstrap_size=1000)


# In[107]:


learner_x._classes


# In[108]:


cate_x


# In[109]:


cate_x_lb


# In[110]:


cate_x_ub


# #### Without Propensity Score Input

# In[111]:


cate_x_no_p, cate_x_lb_no_p, cate_x_ub_no_p = learner_x.fit_predict(X=X, treatment=treatment, y=y, return_ci=True,
                                                     n_bootstraps=100, bootstrap_size=1000)


# In[112]:


learner_x._classes


# In[113]:


cate_x_no_p


# In[114]:


cate_x_lb_no_p


# In[115]:


cate_x_ub_no_p


# ## R-Learner

# ### ATE w/ Confidence Intervals

# #### With Propensity Score Input

# In[116]:


learner_r = BaseRRegressor(XGBRegressor(), control_name='control')
ate_r, ate_r_lb, ate_r_ub = learner_r.estimate_ate(X=X, treatment=treatment, y=y, p=e)


# In[117]:


np.vstack((ate_r_lb, ate_r, ate_r_ub))


# #### Without Propensity Score Input

# In[118]:


learner_r = BaseRRegressor(XGBRegressor(), control_name='control')
ate_r_no_p, ate_r_lb_no_p, ate_r_ub_no_p = learner_r.estimate_ate(X=X, treatment=treatment, y=y)


# In[119]:


np.vstack((ate_r_lb_no_p, ate_r_no_p, ate_r_ub_no_p))


# In[120]:


learner_r.propensity_model


# ### ATE w/ Boostrap Confidence Intervals

# #### With Propensity Score Input

# In[121]:


ate_r_b, ate_r_lb_b, ate_r_ub_b = learner_r.estimate_ate(X=X, treatment=treatment, y=y, p=e, bootstrap_ci=True,
                                                   n_bootstraps=100, bootstrap_size=5000)


# In[122]:


np.vstack((ate_r_lb_b, ate_r_b, ate_r_ub_b))


# #### Without Propensity Score Input

# In[123]:


learner_r = BaseRRegressor(XGBRegressor(), control_name='control')
ate_r_b_no_p, ate_r_lb_b_no_p, ate_r_ub_b_no_p = learner_r.estimate_ate(X=X, treatment=treatment, y=y, bootstrap_ci=True,
                                                   n_bootstraps=100, bootstrap_size=5000)


# In[124]:


np.vstack((ate_r_lb_b_no_p, ate_r_b_no_p, ate_r_ub_b_no_p))


# ### CATE

# #### With Propensity Score Input

# In[125]:


learner_r = BaseRRegressor(XGBRegressor(), control_name='control')
cate_r = learner_r.fit_predict(X=X, treatment=treatment, y=y, p=e)


# In[126]:


cate_r


# #### Without Propensity Score Input

# In[127]:


learner_r = BaseRRegressor(XGBRegressor(), control_name='control')
cate_r_no_p = learner_r.fit_predict(X=X, treatment=treatment, y=y)


# In[128]:


cate_r_no_p


# ### CATE w/ Confidence Intervals

# #### With Propensity Score Input

# In[129]:


learner_r = BaseRRegressor(XGBRegressor(), control_name='control')
cate_r, cate_r_lb, cate_r_ub = learner_r.fit_predict(X=X, treatment=treatment, y=y, p=e, return_ci=True,
                                                     n_bootstraps=100, bootstrap_size=1000)


# In[130]:


cate_r


# In[131]:


cate_r_lb


# In[132]:


cate_r_ub


# #### Without Propensity Score Input

# In[ ]:


learner_r = BaseRRegressor(XGBRegressor(), control_name='control')
cate_r_no_p, cate_r_lb_no_p, cate_r_ub_no_p = learner_r.fit_predict(X=X, treatment=treatment, y=y, p=e, return_ci=True,
                                                     n_bootstraps=100, bootstrap_size=1000)


# In[ ]:


cate_r_no_p


# In[ ]:


cate_r_lb_no_p


# In[ ]:


cate_r_ub_no_p

