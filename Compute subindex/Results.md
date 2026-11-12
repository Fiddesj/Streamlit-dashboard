Dataset Preview (First 5 rows):
  Country_label  Year  GDP_per_capita  Broadband  ...  Investment  R_and_D  Stability    AI_Rate
0   Switzerland  2019           85873  46.912372  ...   50.662526     3.35      100.0  16.170151
1   Switzerland  2020           87530  46.481371  ...   17.808025     3.20      100.0  -4.851695
2   Switzerland  2021           96583  47.943310  ...   61.311288     3.25      100.0  25.896064
3   Switzerland  2022           97809  46.674750  ...   21.824257     3.36      100.0 -24.625428
4   Switzerland  2023          104450  46.836463  ...   56.909131     3.40      100.0  39.235833

[5 rows x 9 columns]
--- Multiple Linear Regression Results ---
C:\Users\Junior\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\base\model.py:1988: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 6, but rank is 5
  res = self.wald_test(
                            OLS Regression Results
==============================================================================
Dep. Variable:                AI_Rate   R-squared:                       0.355
Model:                            OLS   Adj. R-squared:                  0.221
Method:                 Least Squares   F-statistic:                     438.3
Date:                Sun, 25 Oct 2026   Prob (F-statistic):           1.34e-06
Time:                        21:50:34   Log-Likelihood:                -160.27
No. Observations:                  36   AIC:                             334.5
Df Residuals:                      29   BIC:                             345.6
Df Model:                           6
Covariance Type:              cluster
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
const            194.9087     36.124      5.396      0.000     124.107     265.710
GDP_per_capita    -0.0010      0.001     -1.760      0.078      -0.002       0.000
Broadband          0.2120      2.201      0.096      0.923      -4.101       4.525
Electricity       -2.0303      0.613     -3.313      0.001      -3.232      -0.829
Investment         1.1525      0.246      4.688      0.000       0.671       1.634
R_and_D           -4.0845     42.395     -0.096      0.923     -87.177      79.008
Stability          0.5769      0.489      1.180      0.238      -0.381       1.535
==============================================================================
Omnibus:                        5.762   Durbin-Watson:                   1.746
Prob(Omnibus):                  0.056   Jarque-Bera (JB):                4.259
Skew:                           0.735   Prob(JB):                        0.119
Kurtosis:                       3.822   Cond. No.                     1.31e+06
==============================================================================

Notes:
[1] Standard Errors are robust to cluster correlation (CRV1)
[2] The condition number is large, 1.31e+06. This might indicate that there are
strong multicollinearity or other numerical problems.

--- Random Forest (Leave-One-Country-Out CV) ---
LOOCV Overall MAE: 19.54
LOOCV Overall RMSE: 27.48
LOOCV Overall R-squared: -0.13

--- 5-Fold Cross-Validation (80% Train / 20% Val Split) ---
  Fold 1 - MAE: 17.89, RMSE: 19.75, R²: -1.57
  Fold 2 - MAE: 20.64, RMSE: 23.12, R²: 0.06
  Fold 3 - MAE: 20.00, RMSE: 26.78, R²: 0.14
  Fold 4 - MAE: 20.47, RMSE: 26.00, R²: -0.18
  Fold 5 - MAE: 24.20, RMSE: 34.29, R²: -0.08

5-Fold Mean MAE:  20.64 (± 2.04)
5-Fold Mean R²:   -0.33 (± 0.63)

--- Generating SHAP Analysis ---
SHAP Summary Plot saved as 'shap_summary_plot.png'

Feature Importance (Mean Absolute SHAP):
          Feature  Mean_Absolute_SHAP
3      Investment            8.544092
1       Broadband            4.111521
4         R_and_D            1.797754
2     Electricity            1.140955
5       Stability            0.645714
0  GDP_per_capita            0.631057

