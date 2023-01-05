#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pandas as pd, numpy as np
from scipy.stats import norm

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests

import patsy, re
from scipy import linalg
from itertools import chain

# from CTL.causal_tree_learn import CausalTree
from sklearn.model_selection import train_test_split
import plotnine as p
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings('ignore')


# # HTE I: Binary treatment
# 
# Source RMD file: [link](https://docs.google.com/uc?export=download&id=1FSUi4WLfYYKnvWsNWypiQORhkqf5IlFP)
# 
# In the previous chapter, we learned how to estimate the effect of a binary treatment averaged over the entire population. However, the average may obscure important details about how different individuals react to the treatment. In this chapter, we will learn how to estimate the **conditional average treatment effect (CATE)**,
# 
# $$
#   \tau(x) := \mathbf{E}[Y_i(1) - Y_i(0) | X_i = x],
# $$ (cate)
# 
# which is a "localized" version of the average treatment effect conditional on a vector of observable characteristics. 
# 
# It's often the case that {eq}`cate` is too general to be immediately useful, especially when the observable covariates are high-dimensional. It can be hard to estimate reliably without making strong modeling assumptions, and hard to summarize in a useful manner after estimation. In such situations, we will instead try to estimate treatment effect averages for simpler groups
# 
# $$
#   \mathbf{E}[Y_i(1) - Y_i(0) | G_i = g],
# $$ (cate-g)
# 
# where $G_i$ indexes subgroups of interest. Below you'll learn how to estimate and test hypotheses about pre-defined subgroups, and also how to discover subgroups of interest from the data. In this tutorial, you will learn how to use estimates of {eq}`cate` to suggest relevant subgroups $G_i$ (and in the next chapters you will find out other uses of {eq}`cate` in policy learning and evaluation).
# 
# We'll continue using the abridged version of the General Social Survey (GSS) [(Smith, 2016)](https://gss.norc.org/Documents/reports/project-reports/GSSProject%20report32.pdf) dataset that was introduced in the previous chapter. In this dataset, individuals were sent to treatment or control with equal probability, so we are in a randomized setting. However, many of the techniques and code shown below should also work in an observational setting provided that unconfoundedness and overlap are satisfied (these assumptions were defined in the previous chapter).

# As with other chapters in this tutorial, the code below should still work by replacing the next snippet of code with a different dataset, provided that you update the key variables `treatment`, `outcome`, and `covariates` below. Also, please make sure to read the comments as they may be subtle differences depending on whether your dataset was created in a randomized or observational setting.

# In[2]:


data = pd.read_csv( "https://docs.google.com/uc?id=1kSxrVci_EUcSr_Lg1JKk1l7Xd5I9zfRC&export=download" )

n = data.shape[0]

# Treatment: does the the gov't spend too much on "welfare" (1) or "assistance to the poor" (0)
treatment = "w"

# Outcome: 1 for 'yes', 0 for 'no'
outcome = "y"

# Additional covariates
covariates = ["age", "polviews", "income", "educ", "marital", "sex"]


# ## Pre-specified hypotheses
# 
# We will begin by learning how to test pre-specified null hypotheses of the form
# 
# $$
#   H_{0}: \mathbf{E}[Y(1) - Y(0) | G_i = 1] = \mathbf{E}[Y(1) - Y(0) | G_i = 0].
# $$ (hte-hyp)
# 
# That is, that the treatment effect is the same regardless of membership to some group
# $G_i$. Importantly, for now we’ll assume that the group $G_i$ was **pre-specified** -- it was decided _before_ looking at the data.
# 
# In a randomized setting, if the both the treatment  $W_i$ and group membership $G_i$ are binary, we can write
# 
# $$
#   \mathbf{E}[Y_i(W_i)|G_i] = \mathbf{E}[Y_i|W_i, G_i] = \beta_0 + \beta_w W_i + \beta_g G_i + \beta_{wg} W_i G_i
# $$ (linear)
# 
# <font size=1>
# When $W_i$ and $G_i$ are binary, this decomposition is true without loss of generality. Why?
# </font>
# 
# This allows us to write the average effects of $W_i$ and $G_i$ on $Y_i$ as
# 
# $$
#   \begin{aligned}
#     \mathbf{E}[Y(1) | G_i=1] &= \beta_0 + \beta_w W_i + \beta_g G_i + \beta_{wg} W_i G_i, \\
#     \mathbf{E}[Y(1) | G_i=0] &= \beta_0 + \beta_w W_i,  \\
#     \mathbf{E}[Y(0) | G_i=1] &= \beta_0 + \beta_g G_i,  \\
#     \mathbf{E}[Y(0) | G_i=0] &= \beta_0.
#   \end{aligned}
# $$ (decomp)
# 
# Rewriting the null hypothesis {eq}`hte-hyp` in terms of the decomposition {eq}`decomp`, we see that it boils down to a test about the coefficient in the interaction: $\beta_{xw} = 0$. Here’s an example that tests whether the treatment effect is the same for "conservative" (`polviews` < 4) and "liberal" (`polviews` $\geq$ 4) individuals.

# In[3]:


# Only valid in randomized settings

# Suppose this his group was defined prior to collecting the data
data["conservative"] = np.multiply(data.polviews < 4, 1)  # a binary group
group = 'conservative'

# Recall from last chapter -- this is equivalent to running a t-test
fmla = 'y ~ w*conservative'
ols = smf.ols(fmla, data=data).fit(cov_type='HC2')
# print(ols_1.summary())
hypotheses = 'Intercept=0, w=0, conservative=0, w:conservative=0'
t_test = ols.t_test(hypotheses)
print(t_test.summary(xname=list(ols.summary2().tables[1].index)))


# <font size=1>
# Interpret the results above. The coefficient $\beta_{xw}$ is denoted by `w:conservativeTRUE`. Can we detect a difference in treatment effect for conservative vs liberal individuals? For whom is the effect larger?
# </font>
# 
# 
# Sometimes there are many subgroups, leading to multiple hypotheses such as
# 
# $$
# H_0: \mathbf{E}[Y(1) - Y(0) \ | \  G_i = 1] = \mathbf{E}[Y(1) - Y(0) \ | \  G_i = g]
# \qquad
# \text{for many values of }g.
# $$ (mult-hyp)
# 
# In that case, we need to correct for the fact that we are testing for multiple hypotheses, or we will end up with many false positives. The **Bonferroni correction** [(wiki)](https://en.wikipedia.org/wiki/Bonferroni_correction) is a common method for dealing with multiple hypothesis testing, though it is often too conservative to be useful. It is available via the function `p.adjust` from base `R`. The next snippet of code tests whether the treatment effect at each level of `polviews` is different from the treatment effect from `polviews` equals one.

# In[4]:


# Only valid in randomized setting.

# Example: these groups must be defined prior to collecting the data.
group = 'polviews'

# Linear regression.
fmla = 'y ~ w*C(polviews)'
ols = smf.ols(fmla, data=data).fit(cov_type='HC2')

# Retrieve the interaction coefficients
ols_1 = ols.summary2().tables[1].reset_index()
interact = ols_1.loc[ols_1["index"].str.contains("w:")]

hypothesis_1 = []
for i in list(interact['index']):
    hypothesis_1.append(i+str('=0'))
hypotheses = hypothesis_1
t_test = ols.t_test(hypotheses)
# print(t_test.summary(xname=list(interact['index'])))

# Retrieve unadjusted p-values and 
unadj_p_value = list(interact["P>|z|"])
p_adjusted = list(multipletests(unadj_p_value, alpha=0.05, method='bonferroni')[1])

pd.DataFrame(zip(interact["Coef."], interact["Std.Err."], unadj_p_value, p_adjusted),
               columns =['Estimate', 'Std.Err.', 'unadj_p_value', 'adj_p_value'],
            index = list(interact["index"]))


# In[5]:


# Define a function which turn a list or vector-like object into a proper two
# dimensional column vector

def cvec(a):
    """ Turn a list or vector-like object into a proper column vector
    Input
    a: List or vector-like object, has to be a potential input for np.array()
    Output
    vec: two dimensional NumPy array, with the first dimension weakly greater
         than the second (resulting in a column vector for a vector-like input)
    """
    # Conver input into a two dimensional NumPy array
    vec = np.array(a, ndmin=2)

    # Check whether the second dimension is strictly greater than the first
    # (remembering Python's zero indexing)
    if vec.shape[0] < vec.shape[1]:
        # If so, transpose the input vector
        vec = vec.T

    # Return the column vector
    return vec


# In[6]:


def get_cov(X, e, add_intercept=True, homoskedastic=False):
    """ Calculates OLS variance estimator based on X and residuals
    Inputs
    X: n by k matrix, RHS variables
    e: n by 1 vector or vector-like, residuals from an OLS regression
    add_intercept: Boolean, if True, adds an intercept as the first column of X
                   (and increases k by one)
    Outputs
    V_hat: k by k NumPy array, estimated covariance matrix
    """
    # Get the number of observations n and parameters k
    n, k = X.shape

    # Check whether an intercept needs to be added
    if add_intercept:
        # If so, add the intercept
        X = np.concatenate([np.ones(shape=(n,1)), X], axis=1)

        # Don't forget to increase k
        k = k + 1

    # Make sure the residuals are a proper column vector
    e = cvec(e)

    # Calculate X'X
    XX = X.T @ X

    # Calculate its inverse
    XXinv = linalg.inv(XX)

    # Check whether to use homoskedastic errors
    if homoskedastic:
        # If so, calculate the homoskedastic variance estimator
        V_hat = (1 / (n-k)) * XXinv * (e.T @ e)
    else:
        # Otherwise, calculate an intermediate object
        S = (e @ np.ones(shape=(1,k))) * X

        # Then, get the HC0 sandwich estimator
        V_hat = (n / (n-k)) * XXinv @ (S.transpose() @ S) @ XXinv

    # Return the result
    return V_hat


# In[7]:


# Auxiliary function to computes adjusted p-values 
# following the Romano-Wolf method.
# For a reference, see http://ftp.iza.org/dp12845.pdf page 8
#  t.orig: vector of t-statistics from original model
#  t.boot: matrix of t-statistics from bootstrapped models


def romano_wolf_correction(t_orig, t_boot):
    abs_t_orig = np.absolute(t_orig)
    abs_t_boot = np.absolute(t_boot)
    abs_t_sorted = sorted(abs_t_orig, key = float, reverse=True)

    max_order = (-np.array(abs_t_orig)).argsort()
    rev_order = np.argsort(max_order)

    M = t_boot.shape[0]
    S = t_boot.shape[1]

    p_adj = list(np.repeat(0, S))
    p_adj[0] = np.mean(pd.DataFrame(abs_t_boot).apply(np.max, axis=1) > abs_t_sorted[0])

    for s in range(1,S):
        cur_index = max_order[s:S]
        p_init = np.mean(
            pd.DataFrame(abs_t_boot).T.iloc[cur_index].T.apply(np.max, axis=1) > abs_t_sorted[s])
        p_adj[s] = np.max(p_init, p_adj[s])

    aux = []
    for i in rev_order:
        aux.append(p_adj[i])

    p_adj = aux
    
    return(p_adj)


# In[8]:


# Computes adjusted p-values for linear regression (lm) models.
#    model: object of lm class (i.e., a linear reg model)
#    indices: vector of integers for the coefficients that will be tested
#    cov.type: type of standard error (to be passed to sandwich::vcovHC)
#    num.boot: number of null bootstrap samples. Increase to stabilize across runs.
# Note: results are probabilitistic and may change slightly at every run. 
#
# Adapted from the p_adjust from from the hdm package, written by Philipp Bach.
# https://github.com/PhilippBach/hdm_prev/blob/master/R/p_adjust.R

def summary_rw_lm(model_ols, indices='', cov_type="HC2", num_boot=10000):
    # OLS without correction

    # Grab the original t values.
    ols = model_ols
    ols = ols.fit().summary2().tables[1].reset_index()
    summary = ols[ols['index'].isin(list(indices["index"]))].copy()
    t_orig = summary['t']

    # Null resampling.
    # This is a trick to speed up bootstrapping linear models.
    # Here, we don't really need to re-fit linear regressions, which would be a bit slow.
    # We know that betahat ~ N(beta, Sigma), and we have an estimate Sigmahat.
    # So we can approximate "null t-values" by
    #  - Draw beta.boot ~ N(0, Sigma-hat) --- note the 0 here, this is what makes it a *null* t-value.
    #  - Compute t.boot = beta.boot / sqrt(diag(Sigma.hat))

    ols = model_ols.fit(cov_type = cov_type)
    ols_exog = model_ols.exog
    ols_res = model_ols.fit().resid
    Sigma_hat = get_cov(ols_exog, ols_res, add_intercept=False)

    se_orig = pd.Series(np.sqrt(Sigma_hat.diagonal()))

    num_coef = len(se_orig)

    beta_boot = pd.DataFrame(
                np.random.multivariate_normal(
                mean=np.repeat(0, num_coef), cov=Sigma_hat, size=num_boot))

    t_boot = np.array(beta_boot.apply(lambda row: row / se_orig, axis=1))
    ols_1 = ols.summary2().tables[1].reset_index().copy()
    t_boot = t_boot.T[(len(ols_1) - len(t_orig)) : len(ols_1)].T

    p_adj = romano_wolf_correction(t_orig, t_boot)
    result = summary[['index','Coef.','Std.Err.','P>|t|']]
    result.rename(columns={'P>|t|': 'Orig.p-value'}, inplace=True)
    result = result.assign(Adj_p_value = p_adj)

    return(result)


# In[9]:


# This linear regression is only valid in a randomized setting.
fmla = 'y ~ w*C(polviews)'
ols = smf.ols(fmla, data=data)
ols_1 = ols.fit().summary2().tables[1].reset_index()
interact = ols_1.loc[ols_1["index"].str.contains("w:")]

# Applying the romano-wolf correction.
summary_rw_lm(ols, indices=interact)


# <font size=1>
# Compare the adjusted p-values under Romano-Wolf with those obtained via Bonferroni above.
# </font>
# 
# The Bonferroni and Romano-Wolf methods control the **familywise error rate (FWER)**, which is the (asymptotic) probability of rejecting even one true null hypothesis. In other words, for a significance level $\alpha$, they guarantee that with probability $1 - \alpha$ we will make zero false discoveries. However, when the number of hypotheses being tested is very large, this criterion may be so stringent that it prevents us from being able to detect real effects. Instead, there exist alternative procedures that control the (asymptotic) **false discovery rate (FDR)**, defined as the expected proportion of true null hypotheses rejected among all hypotheses rejected. One such procedure is the Benjamini-Hochberg procedure, which is available in base R via `p.adjust(..., method="BH")`. For what remains we'll stick with FWER control, but keep in mind that FDR control can also useful in exploratory analyses or settings in which there's a very large number of hypotheses under consideration.
# 
# As in the previous chapter, when working with observational data under unconfoundedness and overlap, one can use AIPW scores  $\hat{\Gamma}_i$ in place of the raw outcomes  $Y_i$. In the next snippet, we construct AIPW scores using the `causal_forest` function from the `grf` package.

# ## Data-driven hypotheses
# 
# Pre-specifying hypotheses prior to looking at the data is in general good practice to avoid "p-hacking" (e.g., slicing the data into different subgroups until a significant result is found). However, valid tests can also be attained if by **sample splitting**: we can use a subset of the sample to find promising subgroups, then test hypotheses about these subgroups in the remaining sample. This kind of sample splitting for hypothesis testing is called **honesty**.
# 
# ### Via causal trees
# 
# **Causal trees** [(Athey and Imbens)](PNAS, 2016)](https://www.pnas.org/content/pnas/113/27/7353.full.pdf) are an intuitive algorithm that is available in the randomized setting to discover subgroups with different treatment effects.
# 
# At a high level, the idea is to divide the sample into three subsets (not necessarily of equal size). The `splitting` subset is used to fit a decision tree whose objective is modified to maximize heterogeneity in treatment effect estimates across leaves. The `estimation` subset is then used to produce a valid estimate of the treatment effect at each leaf of the fitted tree. Finally, a `test` subset can be used to validate the tree estimates.
# 
# The next snippet uses `honest.causalTree` function from the [`causalTree`](https://github.com/susanathey/causalTree) package. For more details, see the [causalTree documentation](https://github.com/susanathey/causalTree/blob/master/briefintro.pdf).

# In[10]:


import econml
# Main imports
from econml.orf import DMLOrthoForest, DROrthoForest
from econml.dml import CausalForestDML
from econml.grf import CausalForest
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper, WeightedLasso, WeightedLassoCV
from sklearn.linear_model import MultiTaskLassoCV
# Helper imports
import numpy as np
from itertools import product
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from econml.grf import RegressionForest
get_ipython().run_line_magic('matplotlib', 'inline')
import patsy
import seaborn as sns


# In[11]:


from scipy.sparse import csr_matrix
from scipy import sparse


# In[12]:


data = pd.read_csv( "https://docs.google.com/uc?id=1kSxrVci_EUcSr_Lg1JKk1l7Xd5I9zfRC&export=download")
# Convert marital as categorical, and substract 1 to sex
data['sex'] -= 1
data["marital"] = pd.Categorical(data["marital"])
# Define the output and treatment variables.

# Treatment: does the the gov't spend too much on "welfare" (1) or "assistance to the poor" (0)
treatment = "w"

# Outcome: 1 for 'yes', 0 for 'no'
outcome = "y"

# Additional covariates
# drops `marital`
covariates = ["age", "polviews", "income", "educ", "sex"]

# New covariates
covariates = covariates + ["Married", "Widowed", "Divorced", "Separated", "Never_Married"]


# In[13]:


## replace values
marital_val = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never_Married"}
data['marital'] = pd.Categorical(data['marital'].map(marital_val))
# Get new columns
data = pd.get_dummies(data, prefix = "", prefix_sep = "")
## In this case, we remove the marital column and get 
# the columns ["Married", "Widowed", "Divorced", "Separated", "Never_Married"]
data.head(3)


# In[14]:


# Split data into train, estimation, testing 
split_row = np.array_split(list(data.index), 3)

data_train = data.loc[split_row[0]]
data_est = data.loc[split_row[1]]
data_test = data.loc[split_row[2]]


# In[15]:


### econml.dml import CausalForest 
## PlotTree


# In[16]:


# Create a `econ.ml.CausalForest`  object and estimate the model with 
# training data
ctree = CausalForest(
    n_estimators=1, min_samples_leaf=170,
    max_depth=50, inference = False,
    verbose=0, random_state=123,subforest_size=1
    )

ctree.fit(
    data_train[covariates], 
    data_train[treatment], 
    data_train[outcome]
    )


# In[17]:


from sklearn.tree import plot_tree
# Get the trees from the mode
trees = ctree.estimators_

# Plot each tree
for i, tree in enumerate(trees):
    plt.figure()
    plot_tree(tree, feature_names=covariates, filled=True)
    plt.show()
    


# In[18]:


# print the unique values of prediction (numbers of leaf)
tau_hat = ctree.predict(data_est[covariates]).flatten()
leaves = [i for i in range(1,len(np.unique(tau_hat )) + 1 )]
num_leaves = len(np.unique(tau_hat))
ctree.decision_path(data_est[covariates])
print("Number of leaves:", num_leaves)


# In[19]:


# Prediction grouped by each leaf in data_est
data_est = data_est.assign(tau_hat = tau_hat, leaves = pd.Categorical(tau_hat))
data_est["leaves"] = data_est["leaves"].cat.rename_categories(leaves)
data_est.head()


# In[20]:


# This is only valid in randomized datasets.

if num_leaves == 1 :
    
    print("Skipping since there's a single leaf.")
    
elif num_leaves == 2 :
    
    fmla = 'y ~ w*C(leaves)'
    ols = smf.ols(fmla, data=data_est)
    ols_1 = ols.summary2().tables[1].reset_index()
    summary = ols_1.loc[ols_1["index"].str.contains("w:")]
    new_summary = summary[['index','Coef.','Std.Err.','P>|t|']]
    new_summary.rename(columns={'P>|t|': 'Orig.p-value'}, inplace=True)
    
else:
    
# This linear regression is only valid in a randomized setting.

    fmla = 'y ~ w*C(leaves)'
    ols = smf.ols(fmla, data=data_est)
    ols_1 = ols.fit().summary2().tables[1].reset_index()
    interact = ols_1.loc[ols_1["index"].str.contains("w:")]

    # Applying the romano-wolf correction.
    new_summary = summary_rw_lm(ols, indices=interact)

new_summary


# In[21]:


# we can plot the average value for each leaf, in this case, we have 9 leaf
df = pd.DataFrame()

for var_name in covariates:
    form2 =  var_name + " ~ " + "0" + "+" + "C(leaves)"
    ols = smf.ols(formula=form2, data=data_est).fit(cov_type = 'HC2').summary2().tables[1].iloc[:, 0:2]
        
    # Retrieve results
    toget_index = ols["Coef."]
    index = toget_index.index
    cova1 = pd.Series(np.repeat(var_name,num_leaves), index = index, name = "covariate")
    avg = pd.Series(ols["Coef."], name="avg")
    stderr = pd.Series(ols["Std.Err."], name = "stderr")
    leaves = pd.Series(np.arange(1,num_leaves+1), index = index, name = "leaves")
    scaling = pd.Series(norm.cdf((avg - np.mean(avg))/np.std(avg)), index = index, name = "scaling")
    data2 = pd.DataFrame(data=data_est[covariates], columns= covariates)
    variation1= np.std(avg) / np.std(data2[var_name])
    variation = pd.Series(np.repeat(variation1, num_leaves), index = index, name = "variation")
    labels = pd.Series(round(avg,2).astype('str') + "\n" + "(" + round(stderr, 2).astype('str') + ")", index = index, name = "labels")
    
    # Tally up results
    df1 = pd.DataFrame(data = [cova1, avg, stderr, leaves, scaling, variation, labels]).T
    df = df.append(df1)
    
# a small optional trick to ensure heatmap will be in decreasing order of 'variation'
df = df.sort_values(by = ["variation", "covariate"], ascending = False)
df.sample(3)


# In[22]:


df1 = df.pivot(index = "covariate", columns = "leaves", values = ["scaling"]).astype(float)
labels =  df.pivot(index = "covariate", columns = "leaves", values = ["labels"]).to_numpy()

# plot heatmap
ax = plt.subplots(figsize=(10, 10))
ax = sns.heatmap(df1, 
                 annot=labels,
                 annot_kws={"size": 12, 'color':"k"},
                 fmt = '',
                 cmap = "YlGn",
                 linewidths=0,
                 xticklabels = leaves)
plt.tick_params( axis='y', labelsize=15, length=0, labelrotation=0)
plt.tick_params( axis='x', labelsize=15, length=0, labelrotation=0)
plt.xlabel("")
plt.ylabel("")
ax.set_title("Average covariate values within each leaf", fontsize=18, fontweight = "bold")
(ax.set_xticklabels(["Leaf1", "Leaf2", "Leaf3", "Leaf4","Leaf5", "Leaf6","Leaf7", "Leaf8", "Leaf9"], size=15));


# <font size=1>
# Interpret the heatmap above. What describes the subgroups with strongest and weakest estimated treatment effect?
# </font>

# ## Causal Forest 

# The function `causal_forest` from the package `CausalForest` allows us to get estimates of the CATE  (4.1). 

# In[23]:


import econml
# Main imports
from econml.orf import DMLOrthoForest, DROrthoForest
from econml.dml import CausalForestDML
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper, WeightedLasso, WeightedLassoCV
from sklearn.linear_model import MultiTaskLassoCV
# Helper imports
import numpy as np
from itertools import product
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from econml.grf import RegressionForest
get_ipython().run_line_magic('matplotlib', 'inline')
import patsy
import seaborn as sns


# In[24]:


data = pd.read_csv( "https://docs.google.com/uc?id=1kSxrVci_EUcSr_Lg1JKk1l7Xd5I9zfRC&export=download")
# Convert marital as categorical, and substract 1 to sex
data['sex'] -= 1
data["marital"] = pd.Categorical(data["marital"])
# Define the output and treatment variables.

# Treatment: does the the gov't spend too much on "welfare" (1) or "assistance to the poor" (0)
treatment = "w"

# Outcome: 1 for 'yes', 0 for 'no'
outcome = "y"

# Additional covariates
# drops `marital`
covariates = ["age", "polviews", "income", "educ", "sex"]

# New covariates
covariates = covariates + ["Married", "Widowed", "Divorced", "Separated", "Never_Married"]
## replace values
marital_val = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never_Married"}
data['marital'] = pd.Categorical(data['marital'].map(marital_val))
# Get new columns
data = pd.get_dummies(data, prefix = "", prefix_sep = "")
## In this case, we remove the marital column and get 
# the columns ["Married", "Widowed", "Divorced", "Separated", "Never_Married"]
data.head(3)


# In[25]:


# Preparing data to fit a causal forest

T = data.loc[ : ,treatment]
Y = data.loc[ : ,outcome]
X = data[covariates] 

# Estimate a causal forest.
est1 = CausalForest(n_estimators=200, min_samples_leaf=5,
                       max_depth=50,
                       verbose=0, random_state=123)

est1.fit(X, T, Y)

tau_hat = est1.predict(X).flatten()


# In[26]:


# Get residuals  and propensity 
regr = RegressionForest(n_estimators=200, min_samples_leaf=5,
                       max_depth=50,
                       verbose=0, random_state=123)

e_hat = regr.fit(X, T).predict(X).flatten()


# In[27]:


#Propensity score 

Prop = pd.DataFrame({"p_score":e_hat, "Treatment":T})

sns.histplot(data=Prop, x="p_score", hue="Treatment", bins=40, alpha = 0.4)

plt.ylabel('')
plt.xlabel('');


# Having fit a non-parametric method such as a causal forest, a researcher may (incorrectly) start by looking at the distribution of its predictions of the treatment effect. One might be tempted to think: "if the histogram is concentrated at a point, then there is no heterogeneity; if the histogram is spread out, then our estimator has found interesting heterogeneity." However, this may be false.

# In[28]:


# Do not use this for assessing heterogeneity. See text above.
sns.histplot( tau_hat, stat = "percent", bins = 10)
plt.title("CATE estimates");


# If the histogram is concentrated at a point, we may simply be underpowered: our method was not able to detect any heterogeneity, but maybe it would detect it if we had more data. If the histogram is spread out, we may be overfitting: our model is producing very noisy estimates $\widehat{\tau}(x)$, but in fact the true  $\tau(x)$ can be much smoother as a function of $x$.
# 
# The `CausalForest` package also produces a measure of variable importance that indicates how often a variable was used in a tree split. Again, much like the histogram above, this can be a rough diagnostic, but it should not be interpreted as indicating that, for example, variable with low importance is not related to heterogeneity. The reasoning is the same as the one presented in the causal trees section: if two covariates are highly correlated, the trees might split on one covariate but not the other, even though both (or maybe neither) are relevant in the true data-generating process.

# In[29]:


importance = pd.DataFrame({"covariaties" : list(X.columns), "values" : est1.feature_importances()})
importance.sort_values("values", ascending=False)


# #### Data-driven subgroups
# 
# Just as with causal trees, we can use causal forests to divide our observations into subgroups. In place of leaves, we'll rank observation into (say) quintiles according to their estimated CATE prediction; see, e.g., [Chernozhukov, Demirer, Duflo, Fernández-Val (2020)](https://arxiv.org/abs/1712.04802) for similar ideas.
# 
# There's a subtle but important point that needs to be addressed here. As we have mentioned before, when predicting the conditional average treatment effect $\tau(X_i)$ for observation $i$ we should in general avoid using a model that was fitted using observation $i$. This sort of sample splitting (which we called **honesty** above) is one of the required ingredients to get unbiased estimates of the CATE using the methods described here. However, when ranking estimates of two observations $i$ and $j$, we need something a little stronger: we must ensure that the model was not fit using _either_ $i$ _or_ $j$'s data. 
# 
# One way of overcoming this obstacle is simple. First, divide the data into $K$ folds (subsets). Then, cycle through the folds, fitting a CATE model on $K-1$ folds. Next, for each held-out fold, _separately_ rank the unseen observations into $Q$ groups based on their prediction  (i.e., if $Q=5$, then we rank observations by estimated CATE into "top quintile", "second quintile", and so on). After concatenating the independent rankings together, we can study the differences in observations in each rank-group. 
# 
# [This gist](https://gist.github.com/halflearned/bea4e5137c0c81fd18a75f682da466c8) computes the above for `grf`, and it should not be hard to modify it so as to replace forests by any other non-parametric method. However, for `grf` specifically, there's a small trick that allows us to obtain a valid ranking: we can pass a vector of fold indices to the argument `clusters` and rank observations within each fold. This works because estimates for each fold ("cluster")   trees are computed using trees that were not fit using observations from that fold. Here's how to do it. 
# 

# In[30]:


def cluster_causal_forest(X,T,Y, cluster):        
        
        base = pd.concat([Y,T], axis = 1)
        
        for i in range(cluster):
        
            index=range(X.shape[0]) 
            a = np.array_split(np.array(index),cluster)[i]  ## split index
            
            Y = base.drop(base.iloc[list(a),:].index).iloc[:,0]
            T = base.drop(base.iloc[list(a),:].index).iloc[:,1]
            XX = X.drop(X.iloc[list(a),:].index)
            causal = CausalForest(n_estimators=200, min_samples_leaf=5,
                       max_depth=50,
                       verbose=0, random_state=123)
            causal.fit(XX,T,Y)
            
            tau_hat = causal.predict(X=X.iloc[list(a),:]).flatten() # tau(X) estimates using validation data
            vector = i*np.ones( len(list(a)) ) + 1
            globals()[f'data_{i}'] = pd.DataFrame({"tau_hat":tau_hat, "Cluster":vector})
                                                    
        
       
        tau_predict = data_0.copy()

        for k in range(1,cluster):
            tau_predict = tau_predict.append(globals()[f'data_{k}'] , ignore_index=True)
            
        
        
        return tau_predict


# In[31]:


# Valid randomized data and observational data with unconfoundedness+overlap.
# Note: read the comments below carefully. 
# In randomized settings, do not estimate forest.e and e.hat; use known assignment probs.

# Number of rankings that the predictions will be ranking on 
# (e.g., 2 for above/below median estimated CATE, 5 for estimated CATE quintiles, etc.)
num_rankings = 5  

# Prepare for data.splitting
# Assign a fold number to each observation.
# The argument 'clusters' in the next step will mimick K-fold cross-fitting.
num_folds = 5

# Estimate a causal forest.

tau_hat_cluster = cluster_causal_forest(X,T,Y,  cluster = num_folds)

data['ranking'] = np.nan

for i in range(1, num_folds+1):
    split = tau_hat_cluster.Cluster == i
    index = tau_hat_cluster[split].index
    tau_quantile = np.quantile(tau_hat_cluster[split].iloc[:,0], list(np.arange(0,1.1,0.2)))
    labels =[i for i in range(1,6)]
    data.loc[index , ["ranking"]] = pd.cut(tau_hat_cluster[split]["tau_hat"], 
                                               tau_quantile , right=False, labels=labels)


# In[32]:


# Valid only in randomized settings.
# Average difference-in-means within each ranking

# Formula y ~ 0 + ranking + ranking:w
fmla = 'y ~ 0 + C(ranking) + w:C(ranking)'
ols = smf.ols(fmla, data=data).fit(cov_type='HC2')

# Retrieve the interaction coefficients
ols_1 = ols.summary2().tables[1].reset_index()
ols_ate = ols_1.loc[ols_1["index"].str.contains("w:")].iloc[:,1:3]
ols_ate['method'] = "ols"
ols_ate['ranking'] = [f'Q{j}' for j in range(1,6)]
order = [2,3,0,1] # setting column's order
ols_ate = ols_ate[[ols_ate.columns[i] for i in order]]


# In[33]:


ols_ate.rename({'Coef.': 'Estimate', 'Std.Err.': 'se'}, axis=1, inplace = True) 
ols_ate


# In[34]:


# Computing AIPW scores.

regr = RegressionForest(max_depth=50, random_state=123,
                        n_estimators=200)

e_hat = regr.fit(X, T).predict(X).flatten() # P[W=1|X]
m_hat = regr.fit(X, Y).predict(X).flatten() # E[Y|X]


# Estimating mu.hat(X, 1) and mu.hat(X, 0) for obs in held-out sample
# Note: to understand this, read equations 6-8 in this vignette:
# https://grf-labs.github.io/grf/articles/muhats.html
mu_hat0 = m_hat - e_hat * tau_hat        # E[Y|X,W=0] = E[Y|X] - e(X)*tau(X)
mu_hat1 = m_hat + (1 - e_hat) * tau_hat  # E[Y|X,W=1] = E[Y|X] + (1 - e(X))*tau(X)

# AIPW scores
data['aipw_scores'] = tau_hat + T / e_hat * (Y -  mu_hat1) - (1 - T) / (1 - e_hat) * (Y -  mu_hat0)

fmla = 'aipw_scores ~ 0 + C(ranking)'
ols = smf.ols(fmla,data).fit(cov_type='HC2')

# Retrieve the interaction coefficients
ols = ols.summary2().tables[1].reset_index()
forest_ate = ols.loc[ols["index"].str.contains("ranking")].iloc[:,1:3]
forest_ate['method'] = "apiw"
forest_ate['ranking'] = [f'Q{j}' for j in range(1,6)]
order = [2,3,0,1] # setting column's order
forest_ate = forest_ate[[forest_ate.columns[i] for i in order]]
forest_ate.rename({'Coef.': 'Estimate', 'Std.Err.': 'se'}, axis=1, inplace = True) 
forest_ate


# In[35]:


# Concatenate the two results.
res = pd.concat([forest_ate, ols_ate])

# Plotting the point estimate of average treatment effect 
# and 95% confidence intervals around it.
# Define figure, axes, and plot
fig, ax = plt.subplots(figsize=(10, 6))


ax.scatter(x=res['ranking'].unique(), 
         marker='o', s=40, 
         y=res[res.method == "ols"]['Estimate'], color = "blue")

ax.scatter(x=np.arange(0.1,5,1), 
         marker='o', s=40, 
         y=res[res.method == "apiw"]['Estimate'], color = "red")

eb1 = plt.errorbar(x=res['ranking'].unique(), y=res[res.method == "ols"]['Estimate'],
            yerr= 2*res[res.method == "ols"]['se'],
            color = 'darkblue', ls='', capsize = 4)
eb1[-1][0].set_linestyle('--')

eb2 = plt.errorbar(x=np.arange(0.1,5,1), y=res[res.method == "apiw"]['Estimate'],
            yerr= 2*res[res.method == "apiw"]['se'],
            color = 'darkred', ls='', capsize = 4)
eb2[-1][0].set_linestyle('--')

# Set title & labels
plt.title('Average CATE within each ranking (as defined by predicted CATE)',fontsize=15)

# add legend

red_patch = mpatches.Patch(color='red', label='apiw')
blue_patch = mpatches.Patch(color='blue', label='ols')

plt.legend(title = "Method", handles=[blue_patch,red_patch],  bbox_to_anchor=(1.15, 0.7))
    
plt.show()


# When there isn't much detectable heterogeneity, the plot above can end up being non-monotonic. This can mean that the number of observations is too small for us to be able to detect subgroups with relevant differences in treatment effect.
# 
# <font size=1>
# As an exercise, try running the previous two snippets on few data points (e.g., the first thousand observations only). You will likely see the "non-monotonicity" phenomenon just described.
# </font>
# 
# Next, as we did for leaves in a causal tree, we can test e.g., if the prediction for groups 2, 3, etc. are larger than the one in the first group. Here's how to do it based on a difference-in-means estimator. Note the Romano-Wolf multiple-hypothesis testing correction.

# In[36]:


# Valid in randomized settings only.

# y ~ ranking + w + ranking:w

fmla = 'y ~  C(ranking) + w + w:C(ranking)'
ols = smf.ols(fmla, data=data)

# Retrieve the interaction coefficients
ols_1 = ols.fit().summary2().tables[1].reset_index()
interact = ols_1.loc[ols_1["index"].str.contains("w:")]
ols_ate = summary_rw_lm(ols, indices=interact)
ols_ate['ranking'] = [f'Rank {j} - Rank 1' for j in range(2,6)]
order = [5,1,2,3,4] # setting column's order
ols_ate = ols_ate[[ols_ate.columns[i] for i in order]]
ols_ate.rename({'Coef.': 'Estimate', 'Std.Err.': 'se', 'ranking': 'Comparative'}, axis=1, inplace = True) 
ols_ate


# Here's how to do it for AIPW-based estimates, again with Romano-Wolf correction for multiple hypothesis testing. 

# In[37]:


# Valid in randomized and observational settings with unconfoundedness+overlap.

# Using AIPW scores computed above

fmla = 'aipw_scores ~  1 + C(ranking)'
ols = smf.ols(fmla, data=data)

# Retrieve the interaction coefficients
ols_1 = ols.fit().summary2().tables[1].reset_index()
interact = ols_1.loc[1:num_rankings +1 ,:]
forest_ate = summary_rw_lm(ols, indices=interact)
forest_ate['ranking'] = [f'Rank {j} - Rank 1' for j in range(2,6)]
order = [5,1,2,3,4] # setting column's order
forest_ate = forest_ate[[forest_ate.columns[i] for i in order]]
forest_ate.rename({'Coef.': 'Estimate', 'Std.Err.': 'se', 'ranking': 'Comparative'}, axis=1, inplace = True) 
forest_ate



# In[38]:


df = pd.DataFrame()

for var_name in covariates:
    form2 = var_name + " ~ " + "0" + "+" + "C(ranking)"
    ols = smf.ols(formula=form2, data=data).fit(cov_type = 'HC2').summary2().tables[1].iloc[:, 0:2]
    
    
    # Retrieve results
    toget_index = ols["Coef."]
    index = toget_index.index
    cova1 = pd.Series(np.repeat(var_name,num_rankings), index = index, name = "covariate")
    avg = pd.Series(ols["Coef."], name="avg")
    stderr = pd.Series(ols["Std.Err."], name = "stderr")
    ranking = pd.Series(np.arange(1,num_rankings+1), index = index, name = "ranking")
    scaling = pd.Series(norm.cdf((avg - np.mean(avg))/np.std(avg)), index = index, name = "scaling")
    data2 = pd.DataFrame(data=X, columns= covariates)
    variation1= np.std(avg) / np.std(data2[var_name])
    variation = pd.Series(np.repeat(variation1, num_rankings), index = index, name = "variation")
    labels = pd.Series(round(avg,2).astype('str') + "\n" + "(" + round(stderr, 2).astype('str') + ")", index = index, name = "labels")
    
    # Tally up results
    df1 = pd.DataFrame(data = [cova1, avg, stderr, ranking, scaling, variation, labels]).T
    df = df.append(df1)

# a small optional trick to ensure heatmap will be in decreasing order of 'variation'
df = df.sort_values(by = ["variation", "covariate"], ascending = False)

# df = df.iloc[0:(8*num_rankings), :]
df1 = df.pivot(index = "covariate", columns = "ranking", values = ["scaling"]).astype(float)
labels =  df.pivot(index = "covariate", columns = "ranking", values = ["labels"]).to_numpy()

# plot heatmap
ax = plt.subplots(figsize=(10, 10))
ax = sns.heatmap(df1, 
                 annot=labels,
                 annot_kws={"size": 12, 'color':"k"},
                 fmt = '',
                 cmap = "YlGn",
                 linewidths=0,
                 xticklabels = ranking)
plt.tick_params( axis='y', labelsize=15, length=0, labelrotation=0)
plt.tick_params( axis='x', labelsize=15, length=0, labelrotation=0)
plt.xlabel("CATE estimate ranking", fontsize= 10)
plt.ylabel("")
ax.set_title("Average covariate values within group (based on CATE estimate ranking)", fontsize=18, fontweight = "bold")
(ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4","Q5"], size=15));


# #### Best linear projection
# 
# This function provides a doubly robust fit to the linear model $\widehat{\tau}(X_i) = \beta_0 + A_i'\beta_1$, where $A_i$ can be a subset of the covariate columns. The coefficients in this regression are suggestive of general trends, but they should not be interpret as partial effects -- that would only be true if the true model were really linear in covariates, and that's an assumption we shouldn't be willing to make in general

# In[39]:


# dataset for best linear proyection
X['tau_hat'] = tau_hat

# smf.ols('tau_hat ~ age+polviews+income+educ+marital+sex', X).fit(cov_type = 'HC3').summary2().tables[1]
smf.ols("tau_hat ~ " + "+".join(covariates), data = X).fit(cov_type = 'HC3').summary2().tables[1]


# #### Partial dependence
# 
# It may also be interesting to examine how our CATE estimates behave when we change a single covariate, while keeping all the other covariates at a some fixed value. In the plot below we evaluate a variable of interest across quantiles, while keeping all other covariates at their median. 
# 
# It is important to recognize that by keeping some variables at their median we may be evaluating the CATE at $x$ values in regions where there are few or no data points. Also, it may be the case that varying some particular variable while keeping others fixed may just not be very interesting.
# 
# In what follows we'll again use `causal_forest` predictions, along with their variance estimates (set `estimate.variances=TRUE` when predicting to we get estimates of the asymptotic variance of the prediction for each point). Since `grf` predictions are asymptotically normal, we can construct 95\% confidence intervals in the usual manner (i.e., $\hat{\tau}(x) \pm 1.96\sqrt{\widehat{\text{Var}}(\hat{\tau}(x))}$).

# In[40]:


X


# In[41]:


selected_covariate = "polviews"
other_covariates = covariates[:1] + covariates[2:]

# Fitting a forest 
# (commented for convenience; no need re-fit if already fitted above)
# fmla = '0+age+polviews+income+educ+marital+sex'
fmla = "0 + " + "+".join(covariates)

grid_size = 7 


covariate_grid  = np.array_split(data[selected_covariate].sort_values().unique(),grid_size)

# Take median of other covariates 

median = data[other_covariates].median(axis = 0).to_numpy().reshape(1, len(other_covariates))

# duplicates rows 

data_grid = pd.concat([pd.DataFrame(median, columns = other_covariates)]*grid_size)
data_grid[selected_covariate ] = [i[0] for i in covariate_grid]

# Expand the data
X_grid = patsy.dmatrix(fmla, data_grid, return_type = "dataframe")

tau_hat = est1.predict(X=X_grid).flatten() 
tau_hat_ci = est1.predict_interval(X=X_grid)
tau_hat_se  = ((tau_hat_ci[1]-tau_hat_ci[0])/2).flatten()


# In[42]:


fig, ax = plt.subplots(figsize=(10, 6))


ax.scatter(x=data_grid[selected_covariate], 
         marker='o', s=40, 
         y=tau_hat, color = "blue")

ax.plot(data_grid[selected_covariate], tau_hat, color = "blue")

eb1 = plt.errorbar(x=data_grid[selected_covariate], y=tau_hat,
            yerr= tau_hat_se,
            color = 'darkblue', ls='', capsize = 4)
eb1[-1][0].set_linestyle('--')

plt.ylim(-0.6, 0.1)
plt.axhline(y=0, color = 'black').set_linestyle('--')
ax.set_title(f"Predicted treatment effect varying {selected_covariate} (other variables fixed at median)",
             fontsize=11, fontweight = "bold")
(ax.set_xticklabels(range(0,8), size=10));


# In[43]:


selected_covariates = ['polviews','age' ]
other_covariates = covariates[2:]
# Take median of other covariates
median = data[other_covariates].median(axis = 0).to_numpy().reshape(1, 8)

# Compute a grid of values appropriate for the selected covariate
# See other options for constructing grids in the snippet above.

x1_grid = np.array_split(data[selected_covariate].sort_values().unique(),grid_size)
x1_grid_size = len(np.unique(data[selected_covariates[0]]))
x2_grid = np.quantile(data[selected_covariates[1]], list(np.arange(0, 1.01, 1/4)))
x2_grid_size = len(x2_grid)

# duplicates rows 
data_grid = pd.concat([pd.DataFrame(median, columns = other_covariates)]*(x1_grid_size))
data_grid[selected_covariates[0]] =  [i[0] for i in covariate_grid]
data_grid = pd.concat([data_grid]*(x2_grid_size))
data_grid[selected_covariates[1]] =  list(pd.Series(x2_grid).repeat(7))

# Expand the data
X_grid = patsy.dmatrix(fmla, data_grid, return_type = "dataframe")
tau_hat = est1.predict(X = X_grid).flatten()
tau_hat_ci = est1.predict_interval(X = X_grid)
tau_hat_se = ((tau_hat_ci[1]-tau_hat_ci[0])/2).flatten() 


df = X_grid.copy()
df['tau_hat'] = tau_hat
df['tau_hat_se'] = tau_hat_se


# As documented in the original paper on generalized random forests [Athey, Tibshirani and Wager, 2019](https://arxiv.org/abs/1610.01271), the coverage of `grf` confidence intervals can drop if the signal is too dense or too complex relative to the number of observations. Also, to a point, it’s possible to get tighter confidence intervals by increasing the number of trees; see this short vignette for more information.
# 
# We can vary more than one variable at a time.

# In[44]:


# order dataset
df = df.sort_values(by = ["polviews","age"], ascending = [False,True])
labels = np.array(round(df.tau_hat,2).astype('str') + "\n" + "(" + round(df.tau_hat_se, 2).astype('str') + ")")
labels = labels.reshape(x1_grid_size, x2_grid_size) # reshape labels 


# In[45]:


df1 = df.pivot("polviews", "age", "tau_hat")
ax = plt.subplots(figsize=(10, 10))

ax = sns.heatmap(df1,
                 annot=labels,
                 annot_kws={"size": 12, 'color':"k"},
                 fmt = '',
                 cmap = "plasma",
                 linewidths=0,
                )
cbar = ax.collections[0].colorbar
plt.xlabel("Age", fontsize= 10)
plt.ylabel("Polviews", fontsize= 10)
ax.set_title(f"Predicted treatment effect varying {selected_covariates[0]} and{selected_covariates[1]} (other variables fixed at median)",
             fontsize=10, fontweight = "bold");


# ## Further reading
# A readable summary of different method for hypothesis testing correction is laid out in the introduction to [Clarke, Romano and Wolf (2009)](http://ftp.iza.org/dp12845.pdf).
# 
# [Athey and Wager (2019)](https://arxiv.org/abs/1902.07409shows) an application of causal forests to heterogeity analysis in a setting with clustering.
