#!/usr/bin/env python
# coding: utf-8

# # Uplift Tree Visualization

# ## Introduction
# This example notebooks illustrates how to visualize uplift trees for interpretation and diagnosis. 
# 
# #### Supported Models
# These visualization functions work only for tree-based algorithms:
# 
# - Uplift tree/random forests on KL divergence, Euclidean Distance, and Chi-Square
# - Uplift tree/random forests on Contextual Treatment Selection
# 
# Currently, they are NOT supporting Meta-learner algorithms
# 
# - S-learner
# - T-learner
# - X-learner
# - R-learner
# 
# #### Supported Usage
# This notebook will show how to use visualization for:
# 
# - Uplift Tree and Uplift Random Forest
#     - Visualize a trained uplift classification tree model
#     - Visualize an uplift tree in a trained uplift random forests
# 
# - Training and Validation Data
#     - Visualize the validation tree: fill the trained uplift classification tree with validation (or testing) data, and show the statistics for both training data and validation data
#     
# - One Treatment Group and Multiple Treatment Groups
#     - Visualize the case where there are one control group and one treatment group
#     - Visualize the case where there are one control group and multiple treatment groups
# 
# 

# ## Step 1 Load Modules

# ### Load CausalML modules

# In[1]:


from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot


# ### Load standard modules

# In[2]:


import numpy as np
import pandas as pd
from IPython.display import Image
from sklearn.model_selection import train_test_split


# ## One Control + One Treatment for Uplift Classification Tree 

# In[3]:


# Data generation
df, x_names = make_uplift_classification()

# Rename features for easy interpretation of visualization
x_names_new = ['feature_%s'%(i) for i in range(len(x_names))]
rename_dict = {x_names[i]:x_names_new[i] for i in range(len(x_names))}
df = df.rename(columns=rename_dict)
x_names = x_names_new

df.head()

df = df[df['treatment_group_key'].isin(['control','treatment1'])]

# Look at the conversion rate and sample size in each group
df.pivot_table(values='conversion',
               index='treatment_group_key',
               aggfunc=[np.mean, np.size],
               margins=True)


# In[4]:


# Split data to training and testing samples for model validation (next section)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=111)

# Train uplift tree
uplift_model = UpliftTreeClassifier(max_depth = 4, min_samples_leaf = 200, min_samples_treatment = 50, n_reg = 100, evaluationFunction='KL', control_name='control')

uplift_model.fit(df_train[x_names].values,
                 treatment=df_train['treatment_group_key'].values,
                 y=df_train['conversion'].values)


# In[5]:


# Print uplift tree as a string
result = uplift_tree_string(uplift_model.fitted_uplift_tree, x_names)


# #### Read the tree
# - First line: node split condition
# - impurity: the value for the loss function
# - total_sample: total sample size in this node
# - group_sample: sample size by treatment group
# - uplift score: the treatment effect between treatment and control (when there are multiple treatment groups, this is the maximum of the treatment effects)
# - uplift p_value: the p_value for the treatment effect
# - validation uplift score: when validation data is filled in the tree, this reflects the uplift score based on the - validation data. It can be compared with the uplift score (for training data) to check if there are over-fitting issue.

# In[6]:


# Plot uplift tree
graph = uplift_tree_plot(uplift_model.fitted_uplift_tree,x_names)
Image(graph.create_png())


# ### Visualize Validation Tree: One Control + One Treatment for Uplift Classification Tree
# Note the validation uplift score will update.

# In[7]:


### Fill the trained tree with testing data set 
# The uplift score based on testing dataset is shown as validation uplift score in the tree nodes
uplift_model.fill(X=df_test[x_names].values, treatment=df_test['treatment_group_key'].values, y=df_test['conversion'].values)

# Plot uplift tree
graph = uplift_tree_plot(uplift_model.fitted_uplift_tree,x_names)
Image(graph.create_png())


# ### Visualize a Tree in Random Forest

# In[8]:


# Split data to training and testing samples for model validation (next section)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=111)

# Train uplift tree
uplift_model = UpliftRandomForestClassifier(n_estimators=5, max_depth = 5, min_samples_leaf = 200, min_samples_treatment = 50, n_reg = 100, evaluationFunction='KL', control_name='control')

uplift_model.fit(df_train[x_names].values,
                 treatment=df_train['treatment_group_key'].values,
                 y=df_train['conversion'].values)


# In[9]:


# Specify a tree in the random forest (the index can be any integer from 0 to n_estimators-1)
uplift_tree = uplift_model.uplift_forest[0]
# Print uplift tree as a string
result = uplift_tree_string(uplift_tree.fitted_uplift_tree, x_names)


# In[10]:


# Plot uplift tree
graph = uplift_tree_plot(uplift_tree.fitted_uplift_tree,x_names)
Image(graph.create_png())


# #### Fill the tree with validation data

# In[11]:


### Fill the trained tree with testing data set 
# The uplift score based on testing dataset is shown as validation uplift score in the tree nodes
uplift_tree.fill(X=df_test[x_names].values, treatment=df_test['treatment_group_key'].values, y=df_test['conversion'].values)

# Plot uplift tree
graph = uplift_tree_plot(uplift_tree.fitted_uplift_tree,x_names)
Image(graph.create_png())


# ## One Control + Multiple Treatments

# In[12]:


# Data generation
df, x_names = make_uplift_classification()
# Look at the conversion rate and sample size in each group
df.pivot_table(values='conversion',
               index='treatment_group_key',
               aggfunc=[np.mean, np.size],
               margins=True)


# In[13]:


# Split data to training and testing samples for model validation (next section)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=111)

# Train uplift tree
uplift_model = UpliftTreeClassifier(max_depth = 3, min_samples_leaf = 200, min_samples_treatment = 50, n_reg = 100, evaluationFunction='KL', control_name='control')

uplift_model.fit(df_train[x_names].values,
                 treatment=df_train['treatment_group_key'].values,
                 y=df_train['conversion'].values)


# In[14]:


# Plot uplift tree
# The uplift score represents the best uplift score among all treatment effects
graph = uplift_tree_plot(uplift_model.fitted_uplift_tree,x_names)
Image(graph.create_png())


# ### Save the Plot

# In[15]:


# Save the graph as pdf
graph.write_pdf("tbc.pdf")
# Save the graph as png
graph.write_png("tbc.png")

