1. Fixed bug in model.py line 16, Union uses [] brackets, not ().

2. Fixed "bug" of the idea of the Data Scientist that the only important models are "XGBoost with Feature Importance and Class Balance" and "Logistic Regression with Feature Importance and Class Balance",
I "fixed" this misconception, by including both XGBoost and Logistic Regresion models WITHOUT Class Balance to a comparative table,
From there we can see that the percentages for all 4 models are extremely similar to each other, except when it comes for Precision for the XGBoost with Class Balance, where it reaches a 71% precision.

3. Model.py, Removed Union from 'typing' package and removed from 'preprocess' method.

4. Since I first answered the question from 'exploration.ipynb' suggesting to use XGBoost with Feature Importance without Class Balance, as the model having the highest precission, I use this only model inside class 'DelayModel'
 