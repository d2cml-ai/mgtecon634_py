#!/usr/bin/env python
# coding: utf-8

# # Policy Learning I - Binary Treatment
# 
# 
# 
# <font color="#fa8072">Note: this chapter is in 'beta' version and may be edited in the near future.</font>
# 
# A few chapters ago, we learned how to estimate the average effect of a binary treatment (ATE), that is, the value of treating everyone in a population versus treating no one. Once that was established, we asked whether certain subgroups could react differently to the treatment, as we learned how to estimate such heterogeneous treatment effects (HTE). Then, in the previous chapter, we learned how to aggregate these heterogeneous effects to estimate the average outcome that would be attained if treatment assignment were to follow a particular rule, that is, if we were to treat only individuals with certain observable characteristics (policy evaluation). In this chapter, we will learn how to search the space of available treatment rules to approximately _maximize_ the average outcome across the population. That is, we will answer questions of the type: "_who_ should be treated?" We'll call this problem **policy learning**. 
# 
# We'll make a distinction between parametric and non-parametric policies, just as we did with predictive models. Parametric policies are simpler and depend only on a fixed number of parameters, whereas nonparametric policies can increase in complexity with the data. As we'll discuss below, there will be situations in which one or the other will be more appropriate.
# 
# For now, we'll work with the same toy simulation setting that we used in the previous chapter. 
# 

# In[1]:


# Loading relevant packages
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import random 
import time

rest_time = 10 # For time.sleep()

# !pip install patsy
import patsy

## pip install sklearn
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

## !pip install scipy
from scipy.stats import norm, expon, binom
## pip install econml
from econml.grf import RegressionForest, CausalForest, CausalIVForest as instrumental_forest
# from econml.dml import CausalForestDML as causal_forest
from econml.policy import PolicyForest, PolicyTree 
## Users function
from main import *

random.seed(12)
np.random.seed(12)


# In[2]:


## Simulate R, random data
import rpy2.robjects as robjs

r_random_data = robjs.r("""
set.seed(
    1
    , kind = "Mersenne-Twister"
    , normal.kind = "Inversion"
    , sample.kind = "Rejection"
    )
    n_0 <- 1000 * 4

    x_cov <- runif(n_0)
    w <- rbinom(1000, .5, size = 1)
    
    data <- read.csv("https://docs.google.com/uc?id=1kSxrVci_EUcSr_Lg1JKk1l7Xd5I9zfRC&export=download")
    n <- nrow(data)
    data$w <- 1 - data$w

    C <- ifelse(data$w == 1, rexp(n=n, 1/(data$income * data$polviews)), 0)
    
    r_random = list(
        x = x_cov
        , w = w
        , random_cost = C
    )
"""
)


# In[3]:


# Simulating data
n, p, e = 1000, 4, .5 # n: sample size, p : number of covariates, e: binomial probability
x = np.reshape(np.array(r_random_data[0]), (n, p))
w = np.array(r_random_data[1])
y = e * (x[:, 0] - e) + w * (x[:, 1] - e) + .1 * np.random.normal(0, 1, n)

data = pd.DataFrame(x, columns=['x_1', 'x_2', 'x_3', 'x_4'])
data["y"], data['w'] = y, w
outcome, treatment, covariates = "y", "w", list(data.columns)[0:4]


# ## Non-parametric policies
# 
# In the HTE chapter we define the conditional average treatment effect (CATE) function
# 
# $$
#   \tau(x) := \mathop{\mathrm{E}}[Y_i(1) - Y_i(0) | X_i = x],
# $$ (cate)
# 
# that is, the average effect of a binary treatment conditional on observable charateristics. If we knew {eq}`cate`, then a natural policy would be to assigns individuals to treatment their CATE is positive,
# 
# $$
#   \pi^{*} = \mathbb{I}\{\tau(x) \geq 0\}.
# $$
# 
# More generally, if treating that individual costs a known amount $c(x)$, 
# 
# $$
#   \pi^{*} = \mathbb{I}\{\tau(x) \geq c(x)\}.
# $$
# 
# Of course, we don't know {eq}`cate`. However, we can obtain an estimate $\widehat{\tau}(\cdot)$ using any flexible (i.e., non-parametric) method as we learned in the HTE chapter, and then obtain a policy estimate
# 
# $$
#   \hat{\pi}(x) = \mathbb{I}\{ \widehat{\tau}(x) \geq 0\},
# $$
# 
# replacing the zero threshold by some appropriate cost function if needed.
# 
# Once we have an estimated policy, we need to estimate its value. To obtain accurate estimates, we must ensure appropriate **data-splitting**. We cannot estimate and evaluate a policy using the same data set, because that would lead to an overestimate of the value of the policy. One option here is to divide the data into training and test subsets, fit $\widehat{\tau}(\cdot)$ in the training subset and evaluate it in the test subset. This is analogous to what we saw in prediction problems: if we try to evaluate our predictions on the training set, we will overestimate how good our predictions are. 
# 
# The next snippet estimates the conditional treatment effect function via a Lasso model with splines. Note the data splitting.

# In[4]:


# Preparing to run a regression with splines (\\piecewise polynomials).
# Note that if we have a lot of data we should increase the argument `df` below.
# The optimal value of `df` can be found by cross-validation
# i.e., check if the value of the policy, estimated below, increases or decreases as `df` varies. 

# Create the model object
fmla_xw = "y ~ " +  bs_x(0, add = False) + bs_x(1) + bs_x(2) + bs_x(3)
# fmla_xw

# Data-splitting
## Define training and evaluation sets
### Extract rows and cols from dataframe
# train_size = int(.5 * nrow) #Splits works with int type

# Split data 

train_size = .5

data_train, data_test = simple_split(data, train_size)

y, xw = patsy.dmatrices(fmla_xw, data, return_type="dataframe")

y_train, y_test= simple_split(y, train_size)
xw_train, xw_test = simple_split(xw, train_size)

# Fitting the outcome model on the *training* data

model_m = LassoCV(cv = 10, random_state=12)
model_m.fit(xw_train, y_train)
data_0 = data_test.copy()
data_1 = data_test.copy()


# Predict outcome E[Y|X,W=w] for w in {0, 1} on the *test* data

data_0[treatment] = 0
data_1[treatment] = 1

## Construct matirces

y0, xw0 = patsy.dmatrices(fmla_xw, data_0)
y1, xw1 = patsy.dmatrices(fmla_xw, data_1)

# Predict values

mu_hat_1 = model_m.predict(xw1)
mu_hat_0 = model_m.predict(xw0)

# Extract rows 

n_row = data_test.shape[0]

# Computing the CATE estimate tau_hat 
tau_hat = mu_hat_1 - mu_hat_0

# Assignment if tau.hat is positive (or replace by non-zero cost if applicable)
pi_hat = tau_hat > 0

# Estimate assignment probs e(x). 
# (This will be useful for evaluation via AIPW scores a little later)

# In randomized settings assignment probabilities are fixed and known.

e_hat = np.repeat(.5, n_row)


# On the test set, we can evaluate this policy as we learned in the previous chapter. In randomized settings, a simple estimator based on the difference in means is available.
# 
# 

# In[5]:


# Only valid in randomized settings.
a = pi_hat == 1
Y = data_test[outcome]
W = data_test[treatment]

cost = 0 
message_a = "Value estimate: "
message_b = "Std. Error: "

## Extract, value estimate and standard error

c_1 = a & (W == 1)
c_0 = a != 1 & (W == 0)

value_estimate = np.mean(Y[c_1] - cost) * np.mean(a) +                            np.mean(Y[c_0]) * np.mean(a != 1)

value_stderr = np.sqrt(np.var(Y[c_1]) / sum(c_1) * np.mean(a ** 2) +     np.var(Y[c_0]) / sum(c_0) * np.mean(a != 1**2))

print(f"{message_a} {value_estimate}\n{message_b}{value_stderr}")

# extr_val_sd(Y, W, a) # Same results


# In randomized settings and observational settings with unconfoundedness, an estimator of the policy value based on AIPW scores is available. In large samples, it should have smaller variance than the one based on sample averages.

# In[6]:


# Valid in randomized settings and observational settings with unconfoundedness and overlap.
y = data_test[outcome]
w = data_test[treatment]

# AIPW 
gamma_hat_1 = mu_hat_1 + w / e_hat * (y - mu_hat_1)
gamma_hat_0 = mu_hat_0 + (1 - w) / (1 - e_hat) * (y - mu_hat_0)
gamma_hat_pi = pi_hat * gamma_hat_1 + (1 - pi_hat) * gamma_hat_0

## Print the value_estiamte and standard error
ve = np.mean(gamma_hat_pi)
std =  np.std(gamma_hat_pi) / np.sqrt(len(gamma_hat_pi))

print(f"Value estimate: {ve}\nStd.Error: {std}")


# Above we used a flexible linear model, but in fact we can also use any other non-parametric method. The next example uses `econml.grf`. An advantage of using `econ.grf` is that we can leverage [out-of-bag predictions](https://github.com/grf-labs/grf/blob/master/REFERENCE.md#out-of-bag-prediction), so explicit data splitting is not necessary.

# In[7]:


# Using the entire data

x = data[covariates]
y = data[outcome]
w = data[treatment]

# Flexible linear model (econml.grf.Causalforest) 

forest_oob = CausalForest(n_estimators = 100, max_depth = 50, random_state=12)
forest_oob.fit(x, w, y)
# forest_oob = fit_causal_forest(y, w, x)

# Extract residuals
# residuals = forest_oob.residuals_

# Get "out-of-bag" predictions

tau_hat_oob = forest_oob.predict(x).flatten()
pi_hat = tau_hat_oob > 0
# tau_hat_oob


# Again, to evaluate the value of this policy in a randomized setting, we can use the following estimator based on sample averages.
# 
# 

# In[8]:


# Only valid in randomized settings.
# We can use the entire data because predictions are out-of-bag
a = pi_hat == 1

c_1 = a & (w == 1)
c_0 = a != 1 & (w == 0)

value_estimate = np.mean(y[c_1] - cost) * np.mean(a) +                            np.mean(y[c_0]) * np.mean(a != 1)

value_stderr = np.sqrt(np.var(y[c_1]) / sum(c_1) * np.mean(a ** 2) +     np.var(y[c_0]) / sum(c_0) * np.mean(a != 1**2))

print(f"{message_a} {value_estimate}\n{message_b}{value_stderr}")

# Using a Extract function, return, value estimate and standard error
# extr_val_sd(y, w, a)


# And here's how to produce an AIPW-based estimate. Note that that estimates of the propensity scores (`w_hat`) and outcome model (`mu_hat_1`, `mu_hat_0`) are also [out-of-bag](https://github.com/grf-labs/grf/blob/master/REFERENCE.md#out-of-bag-prediction), ensuring appropriate sample splitting.
# 
# 

# In[9]:


# Valid in randomized settings and observational settings with unconfoundedness and overlap.
tau_hat = forest_oob.predict(x).flatten()

# Retrieve relevant quantities.
aux_reg = RegressionForest(random_state = 12, n_estimators = 2000)
e_hat = aux_reg.fit(x, w).predict(x).flatten()
m_hat = aux_reg.fit(x, y).predict(x).flatten()
mu_hat_1 = m_hat + (1 - e_hat) * tau_hat # E[Y|X,W=1] = E[Y|X] + (1 - e(X)) * tau(X) 
mu_hat_0 = m_hat - e_hat * tau_hat # E[Y|X,W=0] = E[Y|X] - e(X) * tau(X)

# ## Compute AIPW score

gamma_hat_1 = mu_hat_1  + w / e_hat * (y - mu_hat_1)
gamma_hat_0 = mu_hat_0 + (1 - w) / (1 - e_hat) * (y - mu_hat_0) # T can be W
gamma_hat_pi = pi_hat * gamma_hat_1 + (1 - pi_hat) * gamma_hat_0

## Value estimates
ve, std = np.mean(gamma_hat_pi), np.std(gamma_hat_pi)/ np.sqrt(len(gamma_hat_pi))
print(f"Value estimate: {ve}\nStd.Error: {std}")


# 
# A technical note. It's easy to get confused and try to "estimate" a nonparametric policy using AIPW scores, as in "$\hat{\pi}(X_i) = \mathbb{I}\{ \widehat{\Gamma}_{i,1} \geq  \widehat{\Gamma}_{i,0} \}$". *This is incorrect*. AIPW scores are very noisy and should never be used "pointwise" like this. They should be used as part of an average (as above), or some other form of aggregation (as we'll see in the next section).
# 
# 
# ## Parametric policies
# 
# In many settings, there are good reasons to constrain the policy to belong to a smaller function class $\pi$. The set $\pi$ may contain only policies that, for example, are transparent and easy to explain to stakeholders, or that are easily implemented in the field. It may also be the case that the set of available policies $\pi$ encodes other desirability criteria, such as satisfying certain budget constraints or depending only on a subset of observable characteristics.
# 
# Estimating such a policy from data is finding an approximate solution to the following constrained maximization problem,
# 
# $$ 
#   \pi^{*} = \arg\max_{\pi \in \Pi} \mathop{\mathrm{E}}[Y(\pi(X_i))].
# $$  (param-pi-oracle)
# 
# 
# Following [Athey and Wager (2020, Econometrica)], we will use the following em\\pirical counterpart of {eq}`param-pi-oracle`,
# 
# $$ 
#   \hat{\pi} = \arg\min_{\pi \in \Pi} \frac{1}{n} \sum_{i=1}^{n} \widehat{\Gamma}_{i,\pi(X_i)}
# $$ (param-pi-problem)
# 
# where $\widehat{\Gamma}_{i,\pi(X_i)}$ are AIPW scores as defined in the previous chapter. As reminder, 
# 
# $$ 
#   \widehat{\Gamma}_{i,\pi(X_i)} = \pi(X_i)\widehat{\Gamma}_{i,1} + (1 - \pi(X_i))\widehat{\Gamma}_{i,0},
# $$ 
# 
# where
# 
# $$  
# \begin{align}
#     \widehat{\Gamma}_{i,1} 
#     &= \hat{\mu}^{-i}(X_i, 1) + \frac{W_i}{\hat{e}^{-i}(X_i)} \left(Y_i -\hat{\mu}^{-i}(X_i, 1)\right), \\
#     \widehat{\Gamma}_{i,0} 
#     &= \hat{\mu}^{-i}(X_i, 0) . \frac{1-W_i}{1-\hat{e}^{-i}(X_i)} \left(Y_i -\hat{\mu}^{-i}(X_i, 0)\right).
# \end{align}
# $$ (aipw)
# 
# Here we use shallow tree policies as our main example of parametric policies. The `R` package `policytree` to find a policy that solves {eq}`param-pi-problem`. In the example below, we'll construct AIPW scores estimated using `grf`, though we could have used any other non-parametric method (with appropriate sample-splitting). See this short [tutorial](https://grf-labs.github.io/policytree/) for other examples using these two packages.
# 
# Let's walk through an example for the data simulated above. The first step is to construct AIPW scores {eq}`aipw`. 
# 

# In[10]:


# Randomized setting: pass the known treatment assignment as an argument.
# Causal Forest Object
forest_prm_p = CausalForest(n_estimators=100, max_depth = 50, random_state=12)
forest_prm_p.fit(x, w, y)

aux_reg_f = RegressionForest(random_state = 12, n_estimators = 2000)

tau_hat = forest_prm_p.predict(x).flatten()

# Retrieve relevant quantiles
e_hat = aux_reg_f.fit(x, w).predict(x).flatten()
m_hat = aux_reg_f.fit(x, y).predict(x).flatten()

mu_hat_1 = m_hat + (1 - e_hat) * tau_hat # E[Y|X,W=1] = E[Y|X] + (1 - e(X)) * tau(X) 
mu_hat_0 = m_hat - e_hat * tau_hat # E[Y|X,W=0] = E[Y|X] - e(X) * tau(X)

gamma_hat_1 = mu_hat_1  + w / e_hat * (y - mu_hat_1)
gamma_hat_0 = mu_hat_0 + (1 - w) / (1 - e_hat) * (y - mu_hat_0)

gamma_mtrx = pd.DataFrame({"gamma_hat_0" : gamma_hat_0,"gamma_hat_1": gamma_hat_1})


# Next, to ensure appropriate sample splitting, we divide our data into training and test subsets. We estimate the policy on the training subset and estimate its value on the test subset.
# 
# 

# In[11]:


# Set train size
train = .5

# Split object into training and testing subset
x_train, x_test = simple_split(x, train)
gamma_mtrx_train, gamma_mtrx_test = simple_split(gamma_mtrx, train)

# Estimate the policy on the training subset 
policy = PolicyTree(max_depth = 2, random_state = 21)        .fit(x_train, gamma_mtrx_train)

## Predict on the test subsets
pi_hat = policy.predict(x_test)
set(pi_hat) # in R return c(2, 1) => pi_hat = predict(policy, x_test) - 1


# 
# We can plot the tree.
# 

# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(25, 5))
policy.plot(treatment_names = ["Control", "Treatment"])
plt.show()


# Note how the treatment rule is rather transparent, in that whether or not each individual is treated depends only on a couple of if-statements. This can be very attractive in settings in which it's important to explain the policy to stakeholders, or reason about its consequences in terms of fairness (e.g., is it okay that these particular subgroups get the treatment?), manipulability (e.g., will individuals lie about their observable characteristics to get a better outcome?), and so on. 
# 
# To evaluate the policy, we again use what we learned in the previous chapter, remembering that we can only use the test set for evaluation. In randomized settings, we can use the following estimator based on sample averages.
# 

# In[13]:


# Only valid for randomized setting!
y_train, y_test = simple_split(y, train)
w_train, w_test = simple_split(w, train)

a = pi_hat == 1

c_1 = a & (w_test == 1)
c_0 = a != 1 & (w_test == 0)

value_estimate = np.mean(y_test[c_1] - cost) * np.mean(a) +                            np.mean(y_test[c_0]) * np.mean(a != 1)

value_stderr = np.sqrt(np.var(y_test[c_1]) / sum(c_1) * np.mean(a ** 2) +     np.var(y_test[c_0]) / sum(c_0) * np.mean(a != 1**2))

print(f"{message_a} {value_estimate}\n{message_b}{value_stderr}")


# Using the remaining AIPW scores produces an estimate that, in large samples, has smaller standard error.
# 
# 

# In[14]:


# Using the remaining AIPW scores produces an estimate that, in large samples, has smaller standard error.
gamma_hat_pi = pi_hat * gamma_mtrx_test.iloc[:, 1] + (1 - pi_hat) * gamma_mtrx_test.iloc[:, 0]
ve, std = np.mean(gamma_hat_pi), np.std(gamma_hat_pi) / np.sqrt(len(gamma_hat_pi))

print(f"Value estimate: {ve}\nStd.Error: {std}")


# A technical note. Very small policy tree leaves make it hard to reliably evaluate policy values, in particular when the treatment is categorical with many levels. You can avoid small tree leaves increasing the `min.node.size` argument in `policy_tree`.
# 
# [Possible edit here: talk about cross-validation?]
# 
# ## Case study
# 
# Let's apply the methods above to our `welfare` dataset, as used in previous chapters.

# In[15]:


# Read in data
data = pd.read_csv("https://docs.google.com/uc?id=1kSxrVci_EUcSr_Lg1JKk1l7Xd5I9zfRC&export=download")

# Extract rows from data
n_row = len(data)

# ## NOTE: invert treatment and control, compared to the ATE and HTE chapters.
data['w'] = 1 - data['w']

# # Treatment is the wording of the question:
# # 'does the the gov't spend too much on 'assistance to the poor' (control: 0)
# # 'does the the gov't spend too much on "welfare"?' (treatment: 1)
treatment = "w"

# # Outcome: 1 for 'yes', 0 for 'no'
outcome = "y"

# # Additional covariates
covariates = ["age", "polviews", "income", "educ", "marital", "sex"]


# It's important to note that there are different types of "heterogeneity" in treatment effects. Sometimes the effect of a treatment is positive throughout, and what changes is the magnitude of the effect. In this case, we would still like to treat everyone. On the other hand, sometimes the treatment effect is positive for certain subgroups and negative for others. The latter is a more interesting scenario for policy learning. 
# 
# In this dataset, however, the effect seems to be mostly positive throughout. That is, i.e., most individuals respond "yes" more often when they are asked about "welfare" than about "assistance to the poor". To make the problem more interesting, we'll artificially modify the problem by introducing a cost of asking about welfare. This is just for illustration here, although there are natural examples in which treatment is indeed costly. Note in the code below how we subtract a cost of `.3` from the AIPW scores associated with treatment.
# 

# In[16]:


# Prepare data
x = data[covariates]
y = data[outcome]
w = data[treatment]

cost = .3

# Fit a policy tree on forest-based AIPW scores
forest_aipw = CausalForest(n_estimators = 100,max_depth=50, random_state=13)
forest_aipw.fit(x, w, y)

tau_hat = forest_aipw.predict(x).flatten()

# Computing AIPW scores
aux_reg = RegressionForest(random_state = 12, n_estimators = 2000)
e_hat = aux_reg.fit(x, w).predict(x).flatten()
m_hat = aux_reg.fit(x, y).predict(x).flatten()

# y_hat = y - residuals[0]
mu_hat_1 = m_hat + (1 - e_hat) * tau_hat # E[Y|X,W=1] = E[Y|X] + (1 - e(X)) * tau(X) 
mu_hat_0 = m_hat - e_hat * tau_hat # E[Y|X,W=0] = E[Y|X] - e(X) * tau(X)

gamma_hat_1 = mu_hat_1  + w / e_hat * (y - mu_hat_1)
gamma_hat_0 = mu_hat_0 + (1 - w) / (1 - e_hat) * (y - mu_hat_0)


### Substracting cost of treatment

gamma_hat_1 -= cost

gamma_mtrx = pd.DataFrame({"gamm_hat_0" : gamma_hat_0, "gamm_hat_1": gamma_hat_1})

# Divide data into train and evaluation sets
train = .8

x_train, x_test = simple_split(x, train)
gamma_mtrx_train, gamma_mtrx_test = simple_split(gamma_mtrx, train)

# Fit policy on training subset

policy = PolicyTree(
    max_depth=2, honest=True, random_state=2
).fit(x_train, gamma_mtrx_train)

# Predicting treatment on test subset
pi_hat = policy.predict(x_test)

# Predicting leaves (useful later)
leaf = policy.apply(x_test)
num_leave = len(set(leaf))
# policy.pre leaf by obsservacion


# In[17]:


plt.figure(figsize=(25, 5))
policy.plot(treatment_names = ["Control", "Treatment"])
plt.show()


# Estimating the value of the learned policy. Note in the code below that we must subtract the cost of treatment.
# 
# 

# In[18]:


a = pi_hat == 1

y_train, y_test = simple_split(y, train)
w_train, w_test = simple_split(w, train)

c_1 = a & (w_test == 1)
c_0 = a != 1 & (w_test == 0)

# Obly valid for randomized setting
# Note the -cost!=0 here!

value_estimate = np.mean(y_test[c_1] - cost) * np.mean(a) +                            np.mean(y_test[c_0]) * np.mean(a != 1)

value_stderr = np.sqrt(np.var(y_test[c_1]) / sum(c_1) * np.mean(a ** 2) +     np.var(y_test[c_0]) / sum(c_0) * np.mean(a != 1**2))

print(f"{message_a} {value_estimate}\n{message_b}{value_stderr}")

### Print Value_estimate [sample avg] 

# extr_val_sd(y_test, w_test, a, cost = cost, message_a="Estimate [Sample avg]", message_b="Std. Error [avg]:")# Declaring the cost of treatment


# In[19]:


# Valid in both randomized and obs setting with unconf + overlap.
### AIPW
### Results from double_robust_score(): gamma_hat_1, gamma_hat_0

gamma_hat_pi = pi_hat * (gamma_mtrx_test.iloc[:, 1]) + (1 - pi_hat) * gamma_mtrx_test.iloc[:, 0]

## Print Estimate [AIPW}
ve, std = np.mean(gamma_hat_pi), np.std(gamma_hat_pi) / np.sqrt(len(gamma_hat_pi))
print(f"Estimate [AIPW]: {ve}\nStd.Error [AIPW]: {std}")


# Testing whether the learned policy value is different from the value attained by the "no-treatment" policy.
# 
# 

# In[20]:


# Only valid for randomized setting.

c_1 = a & (w_test == 1)
c_0 = np.logical_not(a) & (w_test == 0)


diff_estimate = (np.mean(y_test[c_1]) - cost - np.mean(y_test[c_0])) * np.mean(a)
diff_strerr = np.sqrt(np.var(y_test[c_1]) / np.sum(c_1) * np.mean(a)**2 +  np.var(y_test[c_1]) / 
                      np.sum(c_0) * np.mean(a)**2) 

print(f"Difference estimate [sample avg]: {diff_estimate}\t({diff_strerr})")

gamma_hat_pi_diff = gamma_hat_pi - gamma_hat_0
diff_estimate = np.mean(gamma_hat_pi_diff)
diff_strerr = np.std(gamma_hat_pi_diff) / np.sqrt(len(gamma_hat_pi_diff))

print(f"Diference estimate [AIPW]: {diff_estimate}\t ({diff_strerr})")


# ## Topics 1: Subgroups using learned policy
# 
# The learned policy naturally induces interesting subgroups for which we expect the treatment effect to be different. With appropriate sample splitting, we can test that treatment effect is indeed different across "regions" defined by assignment under the learned policy,
# 
# $$
#   H_0: \mathop{\mathrm{E}}[Y_i(1) - Y_i(0)| \hat{\pi}(X_i) = 1] = \mathop{\mathrm{E}}[Y_i(1) - Y_i(0)| \hat{\pi}(X_i) = 0].
# $$
# 
# 

# In[21]:


## Olny from randomized settings

## subset test data

data_test = simple_split(data, train)[1]

# data_test['pi_hat'] = np.array(pi_hat) # pi_hat as covariate
pd.DataFrame(data_test)["pi_hat"] = pi_hat # remove warning
## Formula
fmla = outcome + " ~ 0 + C(pi_hat) + w:C(pi_hat)"

ols = smf.ols(fmla, data=data_test).fit(cov_type='HC2')
ols_coef = ols.summary2().tables[1].reset_index()
ols_coef['Coef.'] = ols_coef['Coef.'] - cost
ols_coef.iloc[2:4, 0:3]


# In[22]:


# Valid in randomized settings and observational settings with unconfoundedness+overlap

gamma_diff = gamma_mtrx_test.iloc[:, 1] -  gamma_mtrx_test.iloc[:, 0]

### gamma_diff as y 
ga_df = pd.DataFrame({'gamma_diff': gamma_diff})
ga_df['pi_hat'] = pi_hat
gam_fml = "gamma_diff ~ 0 + C(pi_hat)"
ols = smf.ols(gam_fml, data = ga_df).fit(cov_type = "HC2").summary2().tables[1].reset_index()
ols


# If we learned a tree policy using the `policytree`, we can test whether treatment effects are different across leaves.
# 
# $$
#   H_0: \mathop{\mathrm{E}}[Y_i(1) - Y_i(0)| \text{Leaf} = 1] = \mathop{\mathrm{E}}[Y_i(1) - Y_i(0)| \text{Leaf} = \ell] \qquad \text{for }\ell \geq 2
# $$

# In[23]:


# Only valid in randomized settings.

fmla = outcome + " ~ + C(leaf) + w:C(leaf)"

pd.DataFrame(data_test)['leaf'] = leaf 
ols = smf.ols(fmla, data=data_test).fit(cov_type = "HC2")
ols_coef = ols.summary2().tables[1].reset_index()
ols_coef.loc[ols_coef["index"].str.contains(":")].iloc[:, 0:3]


# In[24]:


# Valid in randomized settings and observational settings with unconfoundedness+overlap.
ga_df['leaf'] = leaf
fml_gm = "gamma_diff ~ 0 + C(leaf)"
ols = smf.ols(fml_gm, data = ga_df).fit(cov_type = "HC2").summary2().tables[1].reset_index()
ols.iloc[:, 0:3]


# Finally, as we have done in previous chapters, we can check how covariate averages vary across subgroups. This time, the subgroups are defined by treatment assignment under the learned policy.
# 
# $$
#   H_0: \mathop{\mathrm{E}}[X_{ij} | \hat{\pi}(X_i) = 1] = \mathop{\mathrm{E}}[X_{ij} | \hat{\pi}(X_i) = 0] \qquad \text{for each covariate }j
# $$

# In[36]:


df = pd.DataFrame()

for var_name in covariates:
    form2 = var_name + " ~ 0 + C(pi_hat)"
    ols = smf.ols(formula=form2, data=data_test).fit(cov_type = 'HC2').summary2().tables[1].iloc[:, 0:2]
    
    nrow, ncol = ols.shape
    
    # Retrieve results
    toget_index = ols["Coef."]
    index = toget_index.index
    cova1 = pd.Series(np.repeat(var_name,nrow), index = index, name = "covariate")
    avg = pd.Series(ols["Coef."], name="avg")
    stderr = pd.Series(ols["Std.Err."], name = "stderr")
    scaling = pd.Series(norm.cdf((avg - np.mean(avg))/np.std(avg)), index = index, name = "scaling")
    variation1= np.std(avg) / np.std(data[var_name])
    variation = pd.Series(np.repeat(variation1, nrow), index = index, name = "variation")
    labels = pd.Series(round(avg,2).astype('str') + "\n" + "(" + round(stderr, 2).astype('str') + ")", index = index, name = "labels")
    
    # Tally up results
    df1 = pd.DataFrame(data = [cova1, avg, stderr, scaling, variation, labels]).T
    df = df.append(df1)


df["pi_hat"] = ["Control", "Treatment"]*len(covariates) 


df1 = df.pivot("covariate", "pi_hat", "scaling").astype(float).reindex(['polviews', 'educ', 'income', 'sex', 'marital', 'age'])
df1.reindex(['polviews', 'educ', 'income', 'sex', 'marital', 'age'])
labels = df.pivot('covariate', 'pi_hat', 'labels').reindex(['polviews', 'educ', 'income', 'sex', 'marital', 'age']).to_numpy()

ax = plt.subplots(figsize=(10, 10))
ax = sns.heatmap(
    df1, annot=labels,
    annot_kws={"size": 12, 'color':"k"},
    fmt = '',
    cmap = "YlGn",
    linewidths=0
)
plt.tick_params( axis='y', labelsize=15, length=0, labelrotation=0)
plt.tick_params( axis='x', labelsize=15, length=0, labelrotation=0)
plt.xlabel("", fontsize= 10)
plt.ylabel("")
plt.title("Average covariate values within each leaf")


# ## Topics 2: Learning with uncertain costs
# 
# 
# In the previous section, treatment costs were known and (just for simplicity of exposition) fixed across covariate values. However, there are situations in which costs are unknown and must be learned from the data as well. In such situations, we may interested only in policies that do not exceed a certain budget  in expectation.
# 
# Here, we follow [Sun, Du, Wager (2021)](https://arxiv.org/abs/2103.11066) for how to deal with this issue. Their formulation is as follows. In potential outcome notation, each observation can be described by the tuple $(X_i, Y_i(0), Y_i(1), C_i(0), C_i(1))$, where the new pair $(C_i(0), C_i(1))$ represents costs that would be realized if individuals were assigned to control or treatment. Of course, in the data we only observe the tuple $(X_i, W_i, Y_i, C_i)$, where $C_i \equiv C_i(W_i)$. We are interested in approximating the policy $\pi_B^*$ that maximizes the gain from treating relative to not treating anyone while kee\\ping the average relative cost bounded by some known budget $B$,
# 
# $$
#   \\\pi_B^*(x) := \arg\max \mathop{\mathrm{E}}[Y(\pi(X_i))] - \mathop{\mathrm{E}}[Y_i(0)]  \quad \text{such that} \quad \mathop{\mathrm{E}}[C_i(\pi(X_i)) - C_i(0)] \leq B.
# $$
# 
# This paper demonstrates that the optimal policy has the following structure. First, we can order observations in decreasing order according to the following priority ranking,
# 
# $$ 
#   \rho(x) := 
#     \frac{\mathop{\mathrm{E}}[Y_i(1) - Y_i(0) | X_i = x]}
#          {\mathop{\mathrm{E}}[C_i(1) - C_i(0) | X_i = x]}.
# $$ (rho)
# 
# 
# Then, we assign treatment in decreasing order {eq}`rho` until we either treat everyone with positive $\rho(x)$ or the budget is met. The intuition is that individuals for which {eq}`rho` is high have a high expected treatment effect relative to cost, so by assigning them first we obtain a cost-effective policy. We stop once there's no one else for which treatment is expected to be positive or we run out of resources.
# 
# To obtain estimates $\hat{\rho}$ of {eq}`rho` from the data, we have two options. The first is to estimate the numerator $\widehat{\tau}(x) = \mathop{\mathrm{E}}[Y_i(1) - Y_i(0) |X_i = x]$ and the denominator $\hat{\widehat{\Gamma}}(x) = \mathop{\mathrm{E}}[C_i(1) - C_i(0) |X_i = x]$ separately, in a manner analogous to what we saw in the HTE chapter, and compute their ratio, producing the estimate $\hat{\rho}(x) = \widehat{\tau}(x) / \hat{\widehat{\Gamma}}(x)$. We'll see a second option below. 
# 
# Let's put the above into practice. For illustration, we will generate random costs for our data. We'll assume that the costs of treatment are drawn from a conditionally Exponential distribution, and that there are no costs for no treating.

# In[26]:


# Import random costs, and cost to data.
cost = np.array(r_random_data[2])
data['cost'] = cost        


# The next snippet compares two kinds of policies. An "ignore costs" policy which, as the name suggests, orders individuals by $\hat{\tau}$ only without taking costs into account; and the "ratio" policy in which the numerator and denominator of (6.4) are estimated separately. The comparison is made via a **cost curve** that compares the cumulative benefit of treatment with its cumulative cost (both normalized to 1), for all possible budgets at once. More cost-effective policies hug the left corner of the graph more tightly, kee\\ping away from the 45-degree line.
# 
# 

# In[27]:


# Assuming that the assignment probability is known.
# If these are not known, they must be estimated from the data as usual.
e = 0.5  
# Preparing data

y = data[outcome]
w = data[treatment]
x = data[covariates]
d_cost = data['cost']

train = .5

w_hat_train = .5

# Sample splitting. 
# Note that we can't simply rely on out-of-bag observations here.
# train = int(nrow / 2)

x_train, x_test = simple_split(x, train)
y_train, y_test = simple_split(y, train)
w_train, w_test = simple_split(w, train)
c_train, c_test = simple_split(d_cost, train)

# Outcome models
m_forest = RegressionForest(n_estimators=2000, random_state = 12)
m_forest.fit(x_train, c_train)

# time.sleep(10)

# Retrieving forest predictions
y_hat_train = m_forest.predict(x_test).flatten()

## Estimating the numerator
tau_forest = CausalForest(n_estimators = 100, max_depth = 50, random_state=12)
tau_forest.fit(x_train, w_train, y_train)

## Estimating the denominator
gamma_forest = RegressionForest(n_estimators=200)
gamma_forest.fit(x_train, c_train)


### Compute predictions on test set

tau_hat = tau_forest.predict(x_test).flatten()
gamma_hat = gamma_forest.predict(x_test).flatten()


# Rankings

rank_ignore = np.array(tau_hat).argsort()
rank_direct = np.array(tau_hat / gamma_hat).argsort()

n_test = int(nrow - nrow / 2)

# IPW-based estimates of (normalized) treatment and cost
w_hat_test = .5

treatment_ipw = 1 / n_test * (w_test / w_hat_test - (1 - w_test) / (1 - e)) * y_test
cost_ipw = 1 / n_test * w_test / w_hat_test * c_test

# Cumulative benefit and cost of treatment (normalized) for a policy that ignores costs.

t_ipw_ig = np.array(treatment_ipw)[rank_ignore[::-1]]
c_ipw_ig = np.array(cost_ipw)[rank_ignore[::-1]]

treatment_value_ignore = np.cumsum(t_ipw_ig) / np.sum(treatment_ipw)
treatment_cost_ignore = np.cumsum(c_ipw_ig) / np.sum(cost_ipw)

# Cumulative benefit and cost of treatment (normalized) for a policy that uses the ratio, estimated separately.

t_ipw_di = np.array(treatment_ipw)[rank_direct[::-1]]
c_ipw_di = np.array(cost_ipw)[rank_direct[::-1]]

treatment_value_direct = np.cumsum(t_ipw_di) / np.sum(treatment_ipw)
treatment_cost_direct = np.cumsum(c_ipw_di) / np.sum(cost_ipw)


# In[28]:


plt.plot(treatment_cost_ignore, treatment_value_ignore, '#0d5413', label='Ignoring costs')
plt.plot(treatment_cost_direct, treatment_value_direct, '#7c730d', label='Direct Ratio')
plt.title("Cost Curves")
plt.xlabel("(Normalized) cumulative cost")
plt.ylabel("(Normalized) cumulative value")
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 1.5, linestyle = "dotted")
plt.legend()
plt.show()


# To read this graph, we consider a point on the horizontal axis, representing a possible (normalized) budget constraint. At that point, whichever policy is higher is more cost-effective.
# 
# As the authors note, we can also estimate {eq}`rho` in a second way. First, they note that, under overlap and the following extended unconfoudedness assumption
# 
# $$
#   \{Y_i(0), Y_i(1), C_i(1), C_i(0) \perp W_i | X_i \},
# $$
# 
# we can rewrite {eq}`rho` as
# 
# $$
#   \rho(x) := 
#     \frac{\text{Cov}[Y_i, W_i | X_i = x]}
#          {\text{Cov}[C_i, W_i | X_i = x]}.
# $$ (rho-iv)
# 
# As readers with a little more background in causal inference may note, {eq}`rho-iv` coincides with the definition of the conditional local average treatment effect (LATE) if we _were_ to take $W_i$ as an "instrumental variable" and $C_i$ as the "treatment". In fact, instrumental variable methods require different assumptions, so the connection with instrumental variables is tenuous (see the paper for details), but mechanically {eq}`rho-iv` provides us with an estimation procedure: we can use any method used to estimate conditional LATE to produce an estimate $\hat{\rho}$.

# In[29]:


# Estimating rho(x) directly via instrumental forests.
# In observational settings, remove the argument W.hat.
iv_forest = instrumental_forest().fit(x_train, c_train, y_train, Z = w_train)
rho_iv = iv_forest.predict(x_test)
rank_iv = (np.concatenate(rho_iv)).argsort()
# Sorting
t_v_iv = np.array(treatment_ipw)[rank_iv[::-1]]
t_c_iv = np.array(cost_ipw)[rank_iv[::-1]]

treatment_value_iv = np.cumsum(t_v_iv) / np.sum(t_v_iv)
treatment_cost_iv = np.cumsum(t_c_iv) / np.sum(t_c_iv)


# In[30]:


plt.plot(treatment_cost_ignore, treatment_value_ignore, '#0d5413', label='Ignoring costs')
plt.plot(treatment_cost_direct, treatment_value_direct, '#7c730d', label='Direct Ratio')
plt.plot(treatment_cost_iv, treatment_value_iv, "#af1313", label = "Sun. Du. Wager (2021)")
plt.title("Cost Curves")
plt.xlabel("(Normalized) cumulative cost")
plt.ylabel("(Normalized) cumulative value")
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 1.5, linestyle = "dotted")
plt.legend()
plt.show()


# In this example, both the “direct ratio” and the solution based on instrumental forests have similar performance. This isn’t always the case. When the ratio $\rho(x)$ is simpler relative to $\tau(x)$ and $\gamma(x)$, the solution based on instrumental forests may perform better since it is estimating $\rho(x)$ directly, where the “direct ratio” solution needs to estimate the more complicated objects $\tau(x)$ and $\gamma(x)$ separately. At a high level, we should expect $\rho(x)$ to be relatively simpler when there is a strong relationship between $\tau(x)$ and $\gamma(x)$. Here, our simulated costs seem to be somewhat related to CATE (see the plot below), but perhaps not strongly enough to make the instrumental forest solution noticeably better than the one based on ratios.

# In[34]:


# plot
sns.scatterplot(x = gamma_hat, y = tau_hat, alpha=.2)
plt.xlabel("Estimated cost (normalized)")
plt.ylabel("Esimated CATE (normalized)")


# The different policies can be compared by the area between the curves they trace and the 45-degree line, with higher values indicating better policies.
# 
# 

# In[32]:


t_c_i_c = pd.Series(np.array(treatment_cost_ignore))
t_c_r = pd.Series(np.array(treatment_cost_direct))
t_c_iv = pd.Series(np.array(treatment_cost_iv))

##
ignore = np.sum((np.array(treatment_value_ignore) - t_c_i_c) * (t_c_i_c - t_c_i_c.shift(1)))
ratio = np.sum((np.array(treatment_value_direct) - t_c_r) * (t_c_r - t_c_r.shift(1)))
iv = np.sum((np.array(treatment_value_iv) - t_c_iv) * (t_c_iv - t_c_iv.shift(1)))

pd.DataFrame({
    "ignore": ignore,
     "ratio": ratio,
    "iv" : iv
}, index = [0])


# ## Further reading
# 
# The presentation of parametric policies was largely based on Athey and Wager (Econometrica, 2021). A slightly more accessible version of some of the material in the published version can be found in an earlier [ArXiv version](https://arxiv.org/abs/1702.02896v1) of the same paper. Policy comparisons via cost curves can also be found in [Imai and Li (2019)](https://arxiv.org/pdf/1905.05389.pdf).
# 
