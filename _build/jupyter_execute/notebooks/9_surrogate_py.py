#!/usr/bin/env python
# coding: utf-8

# # Athey, S., Chetty, R., Imbens, G. W., & Kang, H. (2019). The surrogate index: Combining short-term proxies to estimate long-term treatment effects more rapidly and precisely (No. w26463). National Bureau of Economic Research.

# [Reference_paper](https://opportunityinsights.org/wp-content/uploads/2019/11/surrogate_paper.pdf)
# 
# [Reference code](https://github.com/OpportunityInsights/Surrogates-Replication-Code)

# ## Setup
# 
# Consider a setting with two samples: an Experimental ($E$) sample and an Observational ($O$)
# sample. The experimental and observational sample contain observations on $N_E$ and $N_O$ units,
# respectively. It is convenient to view the data as consisting of a single sample of size $N =
# N_E + N_O$, with $\mathrm{P, O, E}$ a binary indicator for the group to which unit $i$ belongs.
# 
# For the $N_E$ individuals in the experimental group, there is a single binary treatment of
# interest, $W_i\{0, 1\}$, and a primary outcome, denoted by $Y_i$. This outcome is not observed
# for individuals in the experimental sample. However, we do measure intermediate outcomes,
# which we refer to as surrogates (to be defined precisely in Section 3.2), denoted by $S_i$ for each
# individual. Typically, the surrogate outcomes are vector-valued in order to make the properties
# we define plausible. Finally, we measure pre-treatment covariates $X_i$ for each individual. These
# variables are known not to be affected by the treatment.
# 
# Following the potential outcomes framework or Rubin Causal Model (Rubin 1974; Holland
# 1986; Imbens and Rubin 2015), individuals in this group have two pairs of potential outcomes:
# $\left(Y_i(0), Y_i(1)\right)$ and $\left(S_i(0), S_i(1)\right)$. The realized outcomes are related to their respective potential outcomes as follows.
# 
# 
# $$
# Y_i=\left\{\begin{array}{ll}Y_i(0) & \text { if } W_i=0, \\ Y_i(1) & \text { if } W_i=1,\end{array} \quad\right. \quad \text{and } 
# S_i = \left\{
# \begin{array}{ll}S_i(0) & \text { if } W_i=0, \\ S_i(1) & \text { if } W_i=1,\end{array}
# \right .
# $$
# 
# Overall, the units are characterized by the values of the sextuple $\left(Y_i(0), Y_i(1), S_i(0), S_i(1), X_i, W_i\right)$.
# We do not observe the full sextuple for any units. Rather, for units in the experimental sample
# we observe only the triple $\left(X_i, W_i, S_i\right)$ with support $\mathbb X, \mathbb W = \{0, 1\}$, and $\mathbb S$ respectively. In the
# observational sample, we do not observe to which treatment the NO individuals were assigned.
# We observe the triple $(X_i, S_i, Y_i)$, with support $\mathbb X, \mathbb S$, and $\mathbb Y$ respectively.
# 
# We are interested in the average effect of the treatment on the primary outcome in the
# population from which the experimental sample is drawn:
# 
# $$
# \tau  = \mathbb{E}[
#     y_i(1) - Y_i(0) | P_i = E
# ]
# $$
# 
# 
# or similar estimands, such as the average primary outcome for the treated units, or for some other
# subpopulation. For ease of exposition, we focus on estimating the ATE $\tau$ here. The fundamental
# problem in estimating $\tau$ in the experimental group is that the outcome $Y_i$ is missing for all units in the experimental sample. To address this missing data problem, we exploit the observational
# sample and its link to the experimental sample through the presence of the surrogate outcomes
# $S_i$. Note that the surrogates, like the pre-treatment variables, are not of intrinsic interest. The
# average causal effect of the treatment on the surrogates, $\tau_S = E[S_i(1) − S_i(0)|P_i = \mathrm E]$, is of
# interest only insofar as it aids in estimation of $\tau$.
# 
# ## Identification
# 
# In this section, we discuss three assumptions that together allow us to combine the observational
# and experimental samples and estimate the causal effect of the treatment on the primary outcome
# using a set of intermediate outcomes. The first assumption is unconfoundedness or ignorability,
# common in the program evaluation literature, which ensures that adjusting for pre-treatment
# variables leads to valid causal effects. The second assumption is the surrogacy condition, which
# we define more precisely below, and is the key condition that allows to use the surrogate variables
# to proxy for the primary outcome. The third assumption is comparability, which ensures that
# we can learn about relationships in the experimental sample from the observational sample.
# After stating these three assumptions, we present our main identification result, showing how
# the ATE on the primary outcome can be identified by combining intermediate outcomes under
# these assumptions.
# 
# ### Unconfoundedness 
# 
# For the individuals in the experimental group, define the propensity score as the conditional
# probability of receiving the treatment: $e(x) = \mathrm{pr}\left(Wi = 1 | X_i = x, P_i = E\right)$. We assume that for individuals in the experimental group, treatment assignment is unconfounded and we have overlap in the distribution of pre-treatment variables between the treatment and control groups (Rosenbaum and Rubin 1983).
# 
# 
# 
# **Assumption 1.** (Unconfounded Treatment Assignment / Strong Ignorability)
# 
# $(i)\quad W_i \perp\left(Y_i(0), Y_i(1), S_i(0), S_i(1)\right) \mid X_i, P_i=\mathrm{E}$
# 
# $(ii) \quad 0<e(x)<1$ for all $x \in \mathbb{X}$.
# 
# This assumption implies that in the experimental group, we could estimate the average causal
# effect of the treatment on the outcome $Y_i$ by adjusting for pre-treatment variables, if the $Y_i$ were measured.
# 
# ###  Surrogacy and the Surrogate Index
# 
# Because the primary outcome is not measured in the experimental group, we exploit surrogates
# to identify the treatment effect of $W$ on $Y$ . The defining property of these surrogates $S_i$ is the following condition
# 
# 
# **Assumption 2** (Surrogacy)
# 
# $$
# W_i \perp Y_i \mid S_i, X_i, P_i = \mathrm E.
# $$
# 
# Intuitively, the surrogacy condition requires that the surrogates fully capture the causal
# link between the treatment and the primary outcome. Figure 1 illustrates the content of this
# assumption using directed acyclical graphs to represent the causal chain from the treatment to
# the surrogate to the long-term outcome, as in Pearl (1995). Panel A shows a DAG where the
# surrogacy assumption is satisfied by a single intermediate outcome $S$ that lies on the causal
# chain between $W$ and $Y$ . Panel B shows an example where Assumption 2 is violated because
# there is a direct effect of the treatment on the outcome that does not pass through the surrogate.
# 
# 
# Panel $C$ shows our approach to addressing this problem: introducing multiple intermediate
# outcomes that together span the causal chain from $W$ to $Y$ . In this example, the three inter-
# mediate outcomes together span the causal chain from $W$ to $Y$ and hence can be combined to
# construct a surrogate index that captures long-term treatment effects. Importantly, one does
# not necessarily have to observe every intermediate outcome that lies on the causal chain between
# $W$ and $Y$ . For example, if a treatment (e.g., smaller class sizes) affects earnings by increasing
# both math and science aptitude, math scores by themselves could serve as a valid surrogate if
# math and science scores are perfectly correlated. The key requirement is that the set of inter-
# mediate outcomes together span the set of causal pathways, either because they themselves are
# the causal factors or because they are correlated with the causal factors.
# 
# It is instructive to compare the surrogacy assumption to the exclusion restriction assumption
# familiar to economists in instrumental variables settings. Figure 1d shows a DAG representation
# of the standard instrumental variables (IV) model, where there is an unobserved confounder
# between $W$ and $Y$ . In the standard IV approach, this confound is addressed by introducing
# an instrument $Z$ that affects $W$ but does not affect $Y$ directly (the exclusion restriction). In
# the surrogacy case, we are interested in the effect of $W$ on $Y$ , where we assume there is no
# confounder between $W$ and $Y$ (or, equivalently, we find an instrument for $W$ that eliminates
# such confounds). This is analogous to the (reduced-form) effect of $Z$ on $Y$ in the IV case. The
# reduced-form effect can be estimated directly in the IV case because $Z$ and $Y$ are both observed
# in the same dataset. The problem we address here is how to estimate the effect of $Z$ (or $W$,
# assuming unconfoundedness) on $Y$ when they are not observed in the same dataset. The analog
# to the exclusion restriction here is that there is no direct effect of $W$ on $Y$ that does not run
# through $S$.
# 
# We exploit the availability of multiple intermediate outcomes by defining two concepts: the
# surrogate index and surrogate score.
# 
# 
# #### Deffinition 1: 
# 
# The surrogate Index: *The surrogate index is the conditional expectation
# of the primary outcome given the surrogate outcomes and the pre-treatment variables in the
# observational sample:*
# 
# 
# $$
# h_0(s, x) = \mathbb{E}[Y_i|S_i = s, X_i = x, P_i = O]
# $$
# 
# The surrogate index $h_0(s, x)$ is estimable because we observe the triple $\left(Y_i, S_i, X_i\right)$ in the observational sample. In a linear model, the surrogate index is simply a linear combination of the individual intermediate outcomes – the predicted value from a regression of the primary outcome on the intermediate outcomes.
# 
# #### Definition 2
# 
# The Surrogate Score: *The surrogate score is the conditional probability of
# having received the treatment given the value for the surrogate outcomes and the covariates:*
# 
# $$
# r(s, x) = pr(W_i = 1 | S_i = s, X_i = s, P_i = \mathrm{E})
# $$
# 
# Like the propensity score, the surrogate score facilitates statistical procedures that adjust
# only for scalar differences in other variables, irrespective of the dimension of the statistical
# surrogates.
# 
# #### Proposition 1
# 
# Surrogate Score: *Suppose Assumption 2 holds. Then:*
# 
# $$
# W_i \perp Y_i \mid r(S_i, X_i, P_i) = \mathrm{E}
# $$
# 
# 
# 
# 
# 
# ## Comparibility
# 
# Surrogacy and unconfoundedness by themselves are not sufficient for consistent estimation of $\tau$
# by itself because they do not place restrictions on how the relationship between $Y$ and $S$ in the
# observational sample compares to that in the experimental sample. The final assumption we
# make is that the conditional distribution of $Y_i$ given ($S_i$, $X_i$) in the observational sample is the same as the conditional distribution of $Y_i$ given ($S_i$, $X_i$) in the experimental sample. Formally, Assumption 3. (Comparability of Samples)
# 
# **Assumption 3**. (Comparability of Samples)
# 
# $$
# Y_i\left|S_i, X_i, P_i=\mathrm{O} \sim Y_i\right| S_i, X_i, P_i=\mathrm{E}
# $$
# 
# 
# We can state this assumption equivalently as:
# 
# $$
# P_i \perp Y_i | S_i, X_i
# $$
# 
# To understand the role of the comparability assumption, note that there are two conditional
# expectations that are closely related to the conditional expectation in the definition of the
# surrogate index above, but which we cannot directly estimate because we do not observe $Y$ in
# the experimental sample. The first is the conditional expectation corresponding to the definition
# of the surrogate index above within the experimental sample:
# 
# 
# $$
# h_E(s, x) = \mathbb{E}\left[Y_i | S_i, X_i = x, P_i = E\right]
# $$
# 
# 
# The second is the conditional expectation of the potential outcomes given pre-treatment variables
# and the surrogates:
# 
# $$
# \mu _E(s, x, w) = \mathbb{E}\left[Y_i | S_i = s, X_i = x, W_i = w, P_i = E\right]
# $$
# 
# These conditional expectations are all equivalent under comparability and surrogacy, allowing us
# to take the relationship between $Y$ and $S$ estimated in the observational sample and apply it in
# the experimental sample. In effect, comparability and surrogacy together allow us to impute the
# missing primary outcomes in the experimental sample, as shown by the following proposition.
# 
# #### Proposition 2
# 
# Surrogate Index: 
# 
# *(i) Suppose Assumption 2 holds. Then:*
# 
# $$
# \mu_E(s, x, w) = h_E(s, x), \quad \text{for all } s \in \mathbb{S}, \quad x \in \mathbb{X}, \text{and } w \in \mathbb{W}
# $$
# 
# 
# *(ii) Suppose Assumption 3 holds. Then:*
# 
# $$
# h_E(s, x) = h_O(s, x) \text{ for all } s \in S, \text{ and } x \in X
# $$
# 
# *(iii) Suppose Assumptions 2 and 3 hold. Then:*
# 
# $$
# \mu_E(s, x, w) = h_O(s, x) \text{ for all } s \in S, x \in X, \text{ and } w \in W.
# $$
# 
# 
# Part *(iii)* of Proposition 2 relates the conditional expectation of interest, $\mu_E(s, x, w)$, to a
# conditional expectation that is directly estimable, $h_O(s, x)$.
# Finally, we define weights that make the observational and experimental samples comparable.
# Let $q = N_E/(N_E + N_O)$ denote the sampling weight of being in the experimental sample and
# $(1 − q)$ be the sampling weight of being in the observational sample. Define the propensity to
# be in the experimental sample $P_i = E$ as follows:
# 
# #### Definition 3
# 
# Sampling Score
# 
# $$
# t(s, x) = \mathrm{pr}\left(P_ = E| S_i = s, X_i = x\right) = \frac{
#     \mathrm{pr(S_i = s, X_i = x | P_i = E})q
# }{
#     \mathrm{
#         pr(S_i = s, X_i = x| P_i = E) q + pr(S_i = s, X_i = x | P_i = \mathrm{O})(1 - q)
#     }
# } .
# $$
# 
# 
# We also make the assumption:
# 
# **Assumption 4:** Overlap in Sampling Score
# 
# $$t(s, x) < 1 \text{ for all } x \in \mathbb{S} \text{ and } x \in \mathbb{X}$$
# 
# ### Identification
# 
# 
# We now present our central identification result. We present three different representations of the
# average treatment effect that lead to three estimation strategies. The motivation for developing
# the different representations is that estimators corresponding to those different representations
# can have different properties in finite samples. The first representation requires estimation of
# the surrogate index, but not the surrogate score. The second representation instead requires
# estimation of the surrogate score, but not the surrogate index. The third representation requires
# estimation of both.
# 
# We define the following three objects, all functionals of distributions that are directly estimable from the data, starting with a surrogate index representation:
# 
# $$
# \tau^{\mathrm{E}}=\mathbb{E}\left[h_{\mathrm{O}}\left(S_i, X_i\right) \cdot \frac{W_i}{e\left(X_i\right)}-h_{\mathrm{O}}\left(S_i, X_i\right) \cdot \frac{1-W_i}{1-e\left(X_i\right)} \mid P_i=\mathrm{E}\right]
# $$
# 
# 
# then a surrogate score representation
# 
# $$
# \begin{aligned}
#     \tau^O= & \mathbb{E}\left[Y_i \cdot \frac{r\left(S_i, X_i\right) \cdot t\left(S_i, X_i\right) \cdot(1-q)}{e\left(X_i\right) \cdot\left(1-t\left(S_i, X_i\right)\right) \cdot q}\right.\\
# 
#     & \left. -Y_i \cdot \frac{\left(1-r\left(S_i, X_i\right)\right) \cdot t\left(S_i, X_i\right) \cdot(1-q)}{\left(1-e\left(X_i\right)\right) \cdot\left(1-t\left(S_i, X_i\right)\right) \cdot q} \mid P_i=\mathrm{O}\right]\\
# \end{aligned}
# $$
# 
# and finally an influence function repretentation:
# 
# $$
# \tau^{\mathrm{O}, \mathrm{E}} = \mathbb{E}[
#     \psi (P_i, Y_i, S_i, W_i, X_i)
# ]
# $$
# 
# where
# 
# $$
# \begin{aligned} \psi(p, y, s, w, x) =&\frac{1_{p=\mathrm{E}}}{q}\left(\frac{h_{\mathrm{O}}(s, x) w}{e(x)}-\frac{h_{\mathrm{O}}(s, x)(1-w)}{1-e(x)}\right) \\
# &+ \frac{1_{p=\mathrm{O}}}{1-q}\left(\frac{t(s, x)}{1-t(s, x)} \frac{1-q}{q}\right) \frac{\left(y-h_{\mathrm{O}}(s, x)\right)(r(s, x)-e(x))}{e(x)(1-e(x))} . \end{aligned}
# $$
# 
# Theorem 1. *(Identification) Suppose Assumptions 1–4 hold. Then the average treatment effect is equal to the following three estimable functions of the data:*
# 
# $$
# \tau \equiv \mathbb{E}[
#     Y_i(1) - Y_i(0)|P_i = \mathrm{E} 
# ] = \tau ^ \mathrm{E} = \tau ^ \mathrm{O} =
# \tau ^ {\mathrm{O, E}}.
# $$
# 
# The first representation, $\tau_E$, shows how $\tau$ can be written as the expected value of the
# propensity-score-adjusted difference between treated and controls of the surrogate index in the
# experimental sample. This will lead to an estimation strategy in which the missing $Y_i$ in the experimental sample are imputed by $\hat h(S_i, X_i)$ estimated on the observational sample. The second
# representation, $\tau^\mathrm O$, shows how $\tau$ can be written as the expected value of the difference in two
# weighted averages of the outcome in the observational sample, with the weights a function of
# the surrogate score estimated on the experimental sample and the sampling score. This will lead
# to an estimation strategy in which the $Y_i$ in the observational sample are weighted proportional
# to the estimated surrogate score to estimate $\mathbb E[Y_i(1)|P_i = \mathrm E]$, and weighted proportional to
# one minus the estimated surrogate score to estimate $\mathbb E[Y_i(0)|Pi = E]$. The third representation
# uses the score function representation, requiring estimation of both the surrogate score and the
# surrogate index.
# 
# 
# Under smoothness assumptions, we can derive the semi-parametric efficiency bound for $\tau$
# (e.g., Bickel et al. 1993; Newey 1990). Because the model is just identified (the model has no
# testable implications), it follows that the semi-parametric efficiency bound is the square of the
# influence function $\psi(\cdot) - \tau$:
# 
# 
# $$
# \begin{aligned}
# 
# \mathbb{V}_s  = & \mathbb{E}\left[\left(\psi\left(P_i, Y_i, X_i, S_i, W_i\right)-\tau\right)^2\right] \\
# 
# = &\mathbb{E}\left[\frac{\sigma^2\left(S_i\right)}{1-t\left(S_i, X_i\right)} \cdot\left(\frac{r\left(S_i, X_i\right)}{e\left(X_i\right)^2}+\frac{1-r\left(S_i, X_i\right)}{\left(1-e\left(X_i\right)\right)^2}-2 \cdot \frac{r\left(S_i, X_i\right) \cdot\left(1-r\left(S_i, X_i\right)\right)}{e\left(X_i\right)^2 \cdot\left(1-e\left(X_i\right)\right)^2}\right)\right.\\
# 
# & \left.+\frac{1}{t\left(S_i, X_i\right)} \cdot\left\{\frac{r\left(S_i, X_i\right)}{e\left(X_i\right)} \cdot\left(\mu\left(S_i, X_i\right)-\mu_1\right)^2+\frac{1-r\left(S_i, X_i\right)}{1-e\left(X_i\right)} \cdot\left(\mu\left(S_i, X_i\right)-\mu_0\right)^2\right\}\right]
# \end{aligned}
# $$
# 
# 
# Again because of the just-identified nature of this model, the results in Newey (1994) also imply
# that nonparametric estimators of the surrogacy score, the surrogacy index, and the propensity
# score can be used to obtain efficient estimators for $\tau$

# ## Import modules

# In[1]:


# Data wrangling
import pandas as pd, numpy as np 
# Estimations
import patsy
import statsmodels.formula.api as smf, statsmodels.api as sm
# Plots
import matplotlib.pyplot as plt
# skip irrelevant warnings
import warnings
warnings.filterwarnings('ignore')
from statsmodels.iolib.summary2 import summary_col


# ## Data Wrangling

# In[2]:


# Import Data
dat = pd.read_stata("https://github.com/OpportunityInsights/Surrogates-Replication-Code/raw/master/Data-raw/simulated%20Riverside%20GAIN%20data.dta")
dat.head()


# In[3]:


# To replicate the following stata regression (`reg treatment `outcome'1-`outcome'`q' `), 
## we created the following function
def emp_eq(n, y = "emp_cm36", x_v = "emp"): 
    eq = f"{y} ~ emp1"
    
    def add(x1, x = x_v):
        return f" + {x}{x1}"
    vl = 2
    while vl < n + 1:
        eq = eq + add(vl)
        vl = vl + 1
    
    return eq


# In[4]:


data = dat.copy()

# Generate cumulative means of employment and quarterly wage earnings, from Q6 forward
for i in range(1, 37):
    data["emp_cm1"] = data["emp1"]
    if i > 1:
        data[f"emp_cm{i}"] = (data[f"emp{i}"] + data[f"emp_cm{i - 1}"])
        
for i in range(1, 37):
    data[f"emp_cm{i}"] = data[f"emp_cm{i}"] / i

y_reg = {}
y_reg1 = {}
y_reg2 = {}
bias = {}
surrogate_index = {}

naive_index = {}
emp_weight_p = {}
emp_weight_se = {}

## Store regressions
for i in range(1, 37):
    data_1 = data[data.treatment == 1]
    y_ref, x_ref = patsy.dmatrices(emp_eq(i), data = data_1, return_type="dataframe") 
    y_reg[i] = sm.OLS(y_ref, x_ref).fit()
    
    # emp_values = np.array(pd.DataFrame(y_reg[i].tables[1].data).iloc[1, [1, 2]])
    emp_weight_p[i] = y_reg[i].params 
    emp_weight_se[i] = y_reg[i].conf_int()
    
    ## Estimate treatment effect on surrogate index
    y_nn, x_all = patsy.dmatrices(emp_eq(i), data = data, return_type="dataframe") 
    data['y_surrogate'] = y_reg[i].predict(x_all)
    surrogate_index[i] = smf.ols("y_surrogate ~ C(treatment)", data = data).fit()
    
    y_reg1[i] = smf.ols(f"emp_cm36 ~ emp{i}", data = data_1).fit()
    data[f'single_surrogate{i}'] = y_reg1[i].predict(x_all) 
    naive_index[i] = smf.ols(f"single_surrogate{i} ~ treatment", data = data).fit() 
    
    ## Create naive estimate of treatment effect on mean
    y_reg2[i] = smf.ols(f"emp_cm{i} ~ treatment", data = data).fit()
    
    ## Create "ground truth": experimental estimate of treatment effect on mean 
    exp_reg = smf.ols(f"emp_cm36 ~ treatment", data = data).fit()
    
    ## Bias
    bias[i] = smf.ols(emp_eq(i, "treatment"), data = data).fit()


# ## Tables

# ### Appendix Table 1

# Estimates of Treatment Effects on Employment and Earnings Over Nine Years,
# Varying Quarters of Data Used to Construct Surrogate Index

# In[5]:


for i in [6, 12]:
    ett = np.float_(pd.DataFrame(surrogate_index[i].summary().tables[1].data).iloc[2, [1, 2]])
    print(f"{i} - Quarter Estimate Treatment Effect \t{ett[0]} \n \t\t\t\t\t({ett[1]})")
summary_col([y_reg[6], y_reg[12]], model_names = ("Six - Quarter\nSurrogate Index", "Twelve - Quarter\nSurrogate Index"), regressor_order=("Intercept", "emp1", "emp2", "emp3", "emp4", "emp5", "emp6", "emp7", "emp8", "emp9"))


# ### Appendix Table 2

# Estimates of Treatment Effects on Employment and Earnings Over Nine Years,
# Varying Quarters of Data Used to Construct Surrogate Index

# In[6]:


print(f"Quarter Used \t Employment\t Earnings" )
for i in range(1, 37):
    # print(pd.DataFrame(surrogate_index[i].summary().tables[1].data))
    c_ef = np.float_(pd.DataFrame(surrogate_index[i].summary().tables[1].data).iloc[2, 1])
    s_e = np.float_(pd.DataFrame(surrogate_index[i].summary().tables[1].data).iloc[2, 2])
    print(f"{i} \t\t   {c_ef}\t    .\n\t\t   ({s_e})\t   (.)")


# ## Figures

# In[7]:


# Colours
cl = ["#75b74d", "#226fa5"]
x_lbl = "Quarters Since Random Assignment"


# ### Figure 2: Employment Rates in Riverside GAIN Treatment vs. Control Group, by Quarter

# In[8]:


data = dat.copy()
df = (data.groupby('treatment').mean()* 100).transpose()
df["x"] = range(37)
df = df[df.x > 0]
df.rename(columns = {0: "c", 1:"t"}, inplace = True)
t_m = np.mean(df["t"])
c_m = np.mean(df["c"])
ef = str(round((t_m - c_m), 3))

fig = plt.figure(figsize=(8, 6), dpi = 100, facecolor = 'white')
ax = fig.add_axes([.1, 1, 1, 1])

plt.scatter("x", "t", data = df, c = cl[1], label = "Treatment")
plt.axhline(t_m, color = cl[1], label = "Treatment Mean Over 9 Years")
plt.scatter("x", "c", data = df, c = cl[0], label = "Control", marker = 10)
plt.plot("x", "c", data = df, color = cl[0], label = "")
plt.plot("x", "t", data = df, color = cl[1], label = "")
plt.xlim(0, 36)
plt.ylim(5, 45)
plt.axhline(c_m, color = cl[0], label = "Control Mean Over 9 Years")
plt.legend(loc = (.7, .1))
plt.xlabel(x_lbl)
plt.ylabel("Employment Rate (%)")
plt.yticks(np.arange(10, 41, 10))
plt.xticks(np.arange(1, 37, 5))
plt.annotate(fr"$\tau$ = {ef}%", xy = (36, np.mean([t_m, c_m])), size = 12)
plt.annotate("", xy = (35.7, t_m), xytext = (35.7, c_m), 
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3"))
ax.spines[['top', 'right']].set_visible(False)


# ### Figure 3: Estimates of Treatment Effect on Mean Employment Rates Over Nine Years
# 

# In[9]:


fig, ax = plt.subplots(2, 1, figsize = (10, 12))
for j in range(2):
    for i in exp_reg.conf_int().iloc[1, [0, 1]] * 100:
        ax[j].axhline(i, lw = 1, linestyle = "--")
        ax[j].spines[["top", "right"]].set_visible(False)
        # plt.legend()

for i in range(1, 37):
    # for j in range(2):
    sg = np.array(surrogate_index[i].params[1]) * 100
    nv = np.array(y_reg2[i].params[1]) * 100
    ax[0].scatter(i, sg, color = cl[1], label = "")
    ax[0].scatter(i, nv, color = cl[0], label = "", marker = 10)  
    
ax[0].scatter(1, sg, color = cl[1], label = "Surrogate Index Estimate")
ax[0].scatter(1, nv, color = cl[0], label = "Naive Short-Run Estimate", marker = 10)     
    
for i in range(1, 37):
    ex = naive_index[i].params[1] * 100
    ax[1].scatter(i, ex, color = cl[1], label = "")

ax[1].scatter(1, ex, color = cl[1], label = "Surrogate Estimate Using Emp. Rate in Quarter x Only")


for i in range(2):
    ax[i].axhline(exp_reg.params[1] * 100, label = "Actual Mean Treatment Effect Over 36 Quarters")
    # ax[1].axhline(exp_reg.params[1] * 100, label = "Actual Mean Treatment Effect Over 36 Quarters")
    ax[i].legend(loc = (.2, .1))
    ax[i].set_xlabel(x_lbl)
    ax[i].set_ylabel("Estimated Treatment Effect on Mean \nEmployment Rate Over 9 Years (%)")
    ax[i].set_xticks(np.arange(1, 37, 5))
    ax[i].set_yticks(np.arange(0, 13, 2))
# plt.legend()
ax[1].set_ylim(-1, 9)

fig.subplots_adjust(hspace = .35)
ax[0].set_title("A. Varying Quarters of Data Used to Construct Surrogate Index\n")
ax[1].set_title("B. Using Employment Rate in a Single Quarter as a Surrogate\n")


# ### Figure 4 : Validation of Six-Quarter Surrogate Index: Estimates of Treatment Effects on Mean Employment Rates, Varying Outcome Horizon

# In[10]:


### Estimated treatment effect six tweleve
data = dat.copy()
for i in range(1, 37):
    # print(f"emp_cm{i}")
    data["emp_cm1"] = data["emp1"]
    if i > 1:
        data[f"emp_cm{i}"] = (data[f"emp{i}"] + data[f"emp_cm{i - 1}"])
for i in range(1, 37):
    data[f"emp_cm{i}"] = data[f"emp_cm{i}"] / i


# In[11]:


surr_4 = {}
surr_4_ix = {}
exp_4 = {}

for i in range(6, 37):
    surr_4[i] = smf.ols(emp_eq(6, f"emp_cm{i}"), data = data_1).fit()
    data["surr_4"] = surr_4[i].predict(data)
    surr_4[i] = smf.ols("surr_4 ~ treatment", data = data).fit()
    
    exp_4[i] = smf.ols(f"emp_cm{i} ~ treatment", data = data).fit()
# surr_4[12].summary()

fig = plt.figure(figsize = (8, 6))
ax = fig.add_axes([.1, 1, 1, 1])
lw0 = 1
for i in range(6, 37):
    sr = surr_4[i].params[1] * 100
    sr_ci = surr_4[i].conf_int().iloc[1, [0, 1]] * 100
    
    exp = exp_4[i].params[1] * 100
    exp_ci = exp_4[i].conf_int().iloc[1, [0, 1]] * 100
    
    plt.plot([i, i], sr_ci, color = cl[1], lw = lw0)
    plt.plot([i, i], exp_ci, color = cl[0], lw =lw0)
    plt.scatter(i, exp, c = cl[0], marker=10, label = "")
    plt.scatter(i, sr, c = cl[1], label = "")
    
plt.scatter(36, exp, c = cl[0], label = "Actual Experimental Estimate", marker=10)
plt.scatter(36, sr, c = cl[1], label = "Six-Quarter Surrogate Index Estimate")
plt.xticks(np.arange(6, 37, 5))
plt.yticks(np.arange(4, 15, 2))
plt.legend(loc = (.1, .1))
plt.xlabel(x_lbl)
plt.ylabel("Estimated Treatment Effect on Mean\nEmployment Rate to Quarter x (%)")
ax.spines[['top', 'right']].set_visible(False)


# ### Figure 5: Bounds on Mean Treatment Effect on Employment Rates Over Nine Years, Varying Number of Quarters Used to Construct Surrogate Index

# In[12]:


### Bias
va_r = 0.09500865

bias_01_emp_ar = []
bias_05_emp_ar = []

for i in range(1, 37):
    r2_t = y_reg[i].rsquared
    r2_o = bias[i].rsquared
    bias_01_emp = (va_r * 0.01 * (1 - r2_t) * (1 - r2_o) / va_r) ** (1 / 2) * 100
    bias_05_emp = (va_r * 0.05 * (1 - r2_t) * (1 - r2_o) / va_r) ** (1 / 2) * 100
    bias_01_emp_ar.append(bias_01_emp)
    bias_05_emp_ar.append(bias_05_emp)

s_inx = []
for i in range(1, 37):
    sg = np.array(surrogate_index[i].params[1]) * 100
    s_inx.append(sg)
s_inx = np.array(s_inx)
u1 = s_inx + bias_01_emp_ar
l1 = s_inx - bias_01_emp_ar
u5 = s_inx + bias_05_emp_ar
l5 = s_inx - bias_05_emp_ar
x_q = range(1, 37)


# In[13]:


u1 = s_inx + bias_01_emp_ar
l1 = s_inx - bias_01_emp_ar
u5 = s_inx + bias_05_emp_ar
l5 = s_inx - bias_05_emp_ar
x_q = range(1, 37)


# In[14]:


fig = plt.figure(figsize = (9, 7))
# cl_bias = "#657f91"
ax = fig.add_axes([.1, 1, 1, 1])
plt.axhline(np.mean(s_inx), label = "Actual Mean Treat Eff. Over 36 Quart.", c = "#2b8777")
plt.scatter(x_q, s_inx, label = "Surrogate Index Estimate")
plt.fill_between(x_q, u1, l1, alpha = .3, color = "#404142", label = r"Bounds on Bias: $R^2_Y|W = 1$%")
plt.fill_between(x_q, u5, l5, alpha = .2, color = "#a0afba", label = r"Bounds on Bias: $R^2_Y|W = 5$%")
plt.legend(loc = (.5, .1))
plt.axhline(0, color = "black")
plt.xticks(np.arange(1, 37, 5))
plt.xlabel(x_lbl)
plt.ylabel("Estimated Treatment Effect on Mean\nEmployment Rate Over 9 Years (%")
ax.spines[["top", "right"]].set_visible(False)

