#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This tutorial will introduce key concepts in machine learning-based causal inference. It’s an ongoing project and new chapters will be uploaded as we finish them. A tentative list of topics that will be covered:
# 
# * Introduction to Machine Learning
# * Average Treatment Effects
# * Heterogeneous Treatment Effects (beta)
# * Policy Evaluation (beta)
# * Policy Leaning (beta)
# 
# Please note that this is currently a living document. Chapters marked as “beta” may change substantially and are in most need of feedback. If you find any issues, please feel free to contact Vitor Hadad at vitorh@stanford.edu. The “changelog” below will keep track of major updates and additions.

# ## Getting started

# We’ll illustrate key concepts using snippets of Python code. Each chapter in this tutorial is self-contained. You can download the [repository](https://github.com/alexanderquispe/ml_ci_tutorial/tree/main/Python_tutorials). You should be able to rerun most of the code on a different dataset by editing some of the common variables at the top of the notebook (e.g., data, covariates, and outcome). Exceptions to this rule are marked as comments in the text, so please read them carefully.

# ## Changelog

# * Apr 3, 2021. “Chapter 2: Introduction to Machine Learning” is up.
# * Apr 12, 2021.
#     * Fixed broken link to RMD source. Chapters now have a link at the beginning.
#     * Introduction to ML: Added code to produce colored tables.
# * Apr 13, 2021.
#     * Additional links to RMD source.
#     * Added chapter ATE-1.
# * Apr 16, 2021. Minor fixes to chapter ATE-1.
# * Apr 19, 2021. Removing knitr and kableExtra explicit dependency from Chapter 2. * Fixed other minor typos and inconsistencies in Chapter 2.
# * Apr 26, 2021. Uploading Chapters 3 and 4 (beta versions).
# * May 4, 2021. Added Chapter 5 (beta version).

# ## Acknowledgments

# This tutorial first started as an extension of documents written in part by research assistants, students, and postdocs at the Golub Capital Social Impact Lab. We thank the authors of those documents: Kaleb K. Javier, Niall Keleher, Sylvia Klosin, Nicolaj Søndergaard Mühlbach, Xinkun Nie, and Matt Schaelling. We also thank other people who have been reading this current draft and providing “live” feedback: Cesar Augusto Lopez, Sylvia Klosin, Janelle R. Nelson, Erik Sverdrup.
