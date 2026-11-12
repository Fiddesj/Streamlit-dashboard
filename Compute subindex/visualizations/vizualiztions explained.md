Ran list directory contents
Ran list root directory contents
Analyzed ML%20Model.py:L60-200
Ran check Excel data file existence
Ran verify python libraries · Exited with code 1
Ran check Python 3.13 packages
Checked task Check Python 3.13 packages
Ran check visualizations folder
Created visualize_results.py
Ran run visualize_results.py
Checked task Run visualize_results.py

I have launched the script to generate the visualizations and will notify you as soon as the plots are compiled and saved.

Checked task Run visualize_results.py
Analyzed 1_cv_metrics_barchart.png
Analyzed 2_actual_vs_predicted_loocv.png
Analyzed 3_ols_coefficients_forest_plot.png
Analyzed 4_temporal_trends_performance.png
Analyzed 5_shap_feature_importance.png
Edited visualize_results.py
Edited visualize_results.py
Edited visualize_results.py
Ran re-run visualize_results.py
Checked task Re-run visualize_results.py

I am finishing running the updated visualization script and will present the generated charts and metrics as soon as the task completes.

Checked task Re-run visualize_results.py

I will wait for the execution to finish and share the results.

Checked task Re-run visualize_results.py

I am waiting for the model fitting to finish and will display the output immediately.

Checked task Re-run visualize_results.py

The visualization script is completing its execution in the background; I will share the final plots as soon as it concludes.

Analyzed 2_actual_vs_predicted_loocv.png

To keep your existing pipeline safe and untouched, I created a dedicated visualization script: [**`visualize_results.py`**](file:///c:/Users/Junior/Coding%20Projects/ITDMA%20Coding/Compute%20subindex/visualize_results.py). 

It has been executed and has generated **5 publication-ready figures (300 DPI)** saved in the [**`Compute subindex/visualizations/`**](file:///c:/Users/Junior/Coding%20Projects/ITDMA%20Coding/Compute%20subindex/visualizations/) directory:

---

### Overview of Generated Visualizations

| # | Filename | Type | Purpose & Metric Coverage |
| :---: | :--- | :---: | :--- |
| **1** | [**`1_cv_metrics_barchart.png`**](file:///c:/Users/Junior/Coding%20Projects/ITDMA%20Coding/Compute%20subindex/visualizations/1_cv_metrics_barchart.png) | **Grouped Bar Chart** | **Complements your 5-fold table**: Visualizes fold-by-fold MAE and RMSE with dashed horizontal lines for Mean MAE ($20.64$) and Mean RMSE ($25.99$). |
| **2** | [**`2_actual_vs_predicted_loocv.png`**](file:///c:/Users/Junior/Coding%20Projects/ITDMA%20Coding/Compute%20subindex/visualizations/2_actual_vs_predicted_loocv.png) | **Scatter Plot with $45^\circ$ Line** | **Standard ML performance graph**: Plots actual vs. predicted AI Rate for each country using Leave-One-Country-Out CV, with an inset metric card (MAE: $19.54$, RMSE: $27.48$, $R^2$: $-0.13$). |
| **3** | [**`3_ols_coefficients_forest_plot.png`**](file:///c:/Users/Junior/Coding%20Projects/ITDMA%20Coding/Compute%20subindex/visualizations/3_ols_coefficients_forest_plot.png) | **Forest Plot (Odds / C.I.)** | **Statistical significance**: Plots OLS coefficients with 95% clustered confidence intervals. Visually highlights significant drivers (**Investment** $\beta = 1.15^{***}$, **Electricity** $\beta = -2.03^{**}$) versus variables crossing zero. |
| **4** | [**`4_temporal_trends_performance.png`**](file:///c:/Users/Junior/Coding%20Projects/ITDMA%20Coding/Compute%20subindex/visualizations/4_temporal_trends_performance.png) | **Dual-Panel Line Chart** | **Explicitly covers "Trends"**: <br>• *Left*: Aggregate annual trajectory (2019–2023) of Actual vs. Predicted Mean AI Rate.<br>• *Right*: Individual country volatility over time. |
| **5** | [**`5_shap_feature_importance.png`**](file:///c:/Users/Junior/Coding%20Projects/ITDMA%20Coding/Compute%20subindex/visualizations/5_shap_feature_importance.png) | **Horizontal Bar Chart** | **Model interpretability**: Ranks mean absolute SHAP values (Investment: $8.54$, Broadband: $4.11$, R&D: $1.80$, Electricity: $1.14$, Stability: $0.65$, GDP per capita: $0.63$). |

---

### How to Reference These in Your Document

In your report section (e.g., *Section 7: Results and Insights*):

1. **Under your 5-fold CV table**, insert Figure 1 (`1_cv_metrics_barchart.png`) to give an immediate visual impression of fold stability.
2. **For model predictive accuracy**, insert Figure 2 (`2_actual_vs_predicted_loocv.png`) to explain out-of-sample generalization.
3. **For econometric & causal interpretation**, insert Figure 3 (`3_ols_coefficients_forest_plot.png`) and Figure 5 (`5_shap_feature_importance.png`) to discuss which features most strongly impact AI adoption.
4. **For temporal trends**, insert Figure 4 (`4_temporal_trends_performance.png`) to illustrate the pre/post-2022 dynamics.

You can re-run this script anytime with:
```powershell
python "Compute subindex\visualize_results.py"
```