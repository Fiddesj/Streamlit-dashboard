"""
Visualizations for Compute Subindex ML Model Results
Generates presentation- and publication-ready charts and graphs 
highlighting performance metrics, trends, and feature impacts.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Ensure target output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(output_dir, exist_ok=True)

# 1. DATA PREPARATION (matching ML Model.py)
excel_path = r"C:\Users\Junior\Downloads\AI_Research_Data_Master (1) (1).xlsx"
features = ['GDP_per_capita', 'Broadband', 'Electricity', 'Investment', 'R_and_D', 'Stability']
target = 'AI_Rate'
group_col = 'Country_label'

def load_data():
    df_pol = pd.read_excel(excel_path, sheet_name='Political_Analysis')
    df_pol['Country'] = df_pol['Country'].astype(str).str.strip().replace({'United States': 'USA'})
    df_pol['Year'] = pd.to_numeric(df_pol['Year'], errors='coerce')
    pol_sub = df_pol[['Country', 'Year', 'Stability_Index']].rename(columns={'Stability_Index': 'Stability'})

    df_dig = pd.read_excel(excel_path, sheet_name='Digital Infrastructure Score')
    df_dig['Country'] = df_dig['Country'].astype(str).str.strip().replace({'United States': 'USA'})
    df_dig['Year'] = pd.to_numeric(df_dig['Year'], errors='coerce')
    elec_col = [c for c in df_dig.columns if 'elec' in str(c).lower()][0]
    broad_col = [c for c in df_dig.columns if 'broadband' in str(c).lower()][0]
    dig_sub = df_dig[['Country', 'Year', elec_col, broad_col]].rename(columns={
        elec_col: 'Electricity',
        broad_col: 'Broadband'
    })

    df_pred = pd.read_excel(excel_path, sheet_name='1_Predictor_Data', header=3)
    df_pred['Country'] = df_pred['Country'].astype(str).str.strip().replace({'United States': 'USA'})
    df_pred['Year'] = pd.to_numeric(df_pred['Year'], errors='coerce')
    gdp_col = [c for c in df_pred.columns if 'gdp per capita' in str(c).lower()][0]
    rd_col = [c for c in df_pred.columns if 'r&d' in str(c).lower()][0]
    pred_sub = df_pred[['Country', 'Year', gdp_col, rd_col]].rename(columns={
        gdp_col: 'GDP_per_capita',
        rd_col: 'R_and_D'
    })

    df_adi = pd.read_excel(excel_path, sheet_name='3_ADI_Calculation', header=3)
    df_adi['Country'] = df_adi['Country'].astype(str).str.strip().replace({'United States': 'USA'})
    df_adi['Year'] = pd.to_numeric(df_adi['Year'], errors='coerce')
    inv_col = [c for c in df_adi.columns if 'investment' in str(c).lower() and 'sub-index' in str(c).lower()][0]
    target_col = [c for c in df_adi.columns if 'rate' in str(c).lower() or 'adr' in str(c).lower()][0]
    adi_sub = df_adi[['Country', 'Year', inv_col, target_col]].rename(columns={
        inv_col: 'Investment',
        target_col: 'AI_Rate'
    })

    df = pred_sub.merge(dig_sub, on=['Country', 'Year'], how='inner')
    df = df.merge(pol_sub, on=['Country', 'Year'], how='inner')
    df = df.merge(adi_sub, on=['Country', 'Year'], how='inner')
    df['Country_label'] = df['Country']

    numeric_cols = ['GDP_per_capita', 'Broadband', 'Electricity', 'Investment', 'R_and_D', 'Stability', 'AI_Rate']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['R_and_D'] = df.groupby('Country')['R_and_D'].ffill()
    df = df.dropna(subset=features + [target, group_col]).reset_index(drop=True)
    return df

# -------------------------------------------------------------
# GRAPH 1: 5-Fold Cross-Validation Metrics Bar Chart
# -------------------------------------------------------------
def plot_cv_metrics():
    folds = [f"Fold {i}" for i in range(1, 6)]
    mae = [17.89, 20.64, 20.00, 20.47, 24.20]
    rmse = [19.75, 23.12, 26.78, 26.00, 34.29]
    r2 = [-1.57, 0.06, 0.14, -0.18, -0.08]

    x = np.arange(len(folds))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    # Bars for MAE and RMSE
    rects1 = ax1.bar(x - width/2, mae, width, label='MAE', color='#2b5c8f', edgecolor='black', linewidth=0.6)
    rects2 = ax1.bar(x + width/2, rmse, width, label='RMSE', color='#e26d5c', edgecolor='black', linewidth=0.6)

    # Reference mean lines
    mean_mae = np.mean(mae)
    mean_rmse = np.mean(rmse)
    ax1.axhline(mean_mae, color='#1e3d59', linestyle='--', linewidth=1.2, label=f'Mean MAE ({mean_mae:.2f})')
    ax1.axhline(mean_rmse, color='#b23b2a', linestyle='--', linewidth=1.2, label=f'Mean RMSE ({mean_rmse:.2f})')

    # Value labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Error Magnitude', fontsize=11, fontweight='bold')
    ax1.set_title('Cross-Validation Error Metrics Across 5 Folds (Random Forest)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds, fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 40)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    plt.tight_layout()

    file_path = os.path.join(output_dir, "1_cv_metrics_barchart.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_path}")

def get_loocv_predictions(df):
    """Generates exact LOOCV predictions using the grid search tuned model from ML Model.py."""
    X = df[features]
    y = df[target]
    groups = df[group_col]

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [2, 3, 4],
        'min_samples_leaf': [2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    logo = LeaveOneGroupOut()
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error'
    )

    y_pred = np.zeros(len(df))
    for train_idx, test_idx in logo.split(X, y, groups):
        grid_search.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred[test_idx] = grid_search.best_estimator_.predict(X.iloc[test_idx])

    return y.values, y_pred

# -------------------------------------------------------------
# GRAPH 2: Actual vs. Predicted AI Rate (Scatter with 1:1 Line)
# -------------------------------------------------------------
def plot_actual_vs_predicted(df):
    y_true, y_pred = get_loocv_predictions(df)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)

    # Scatter plot by country
    for country in df['Country'].unique():
        mask = (df['Country'] == country).values
        ax.scatter(y_true[mask], y_pred[mask], label=country, s=70, alpha=0.85, edgecolors='black', linewidth=0.6)

    # Perfect prediction 1:1 line
    lims = [min(min(y_true), min(y_pred)) - 5, max(max(y_true), max(y_pred)) + 5]
    ax.plot(lims, lims, color='gray', linestyle='--', linewidth=1.5, label='Ideal 1:1 Reference')

    # Trend line (regression line through predictions)
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_vals = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_vals, p(x_vals), color='#1e3d59', linewidth=1.8, label=f'Model Fit (slope={z[0]:.2f})')

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual AI Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted AI Rate (LOOCV)', fontsize=11, fontweight='bold')
    ax.set_title('Random Forest LOOCV: Actual vs. Predicted AI Rate', fontsize=13, fontweight='bold', pad=12)

    # Annotation box for summary metrics (matching Results.md)
    metric_text = f"LOOCV Overall Metrics:\nMAE:  {mae:.2f}\nRMSE: {rmse:.2f}\nR²:   {r2:.2f}"
    ax.text(0.05, 0.95, metric_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#b0b0b0', alpha=0.9),
            fontsize=10, family='monospace')

    ax.legend(loc='lower right', frameon=True, fontsize=9)
    plt.tight_layout()

    file_path = os.path.join(output_dir, "2_actual_vs_predicted_loocv.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_path}")

# -------------------------------------------------------------
# GRAPH 3: OLS Regression Coefficients & 95% Confidence Intervals
# -------------------------------------------------------------
def plot_ols_coefficients():
    # Coefs, Std Errors, CI from Results.md
    var_names = ['Investment', 'Electricity', 'Stability', 'Broadband', 'GDP_per_capita', 'R_and_D']
    display_names = ['Investment', 'Electricity', 'Stability', 'Broadband', 'GDP per capita', 'R&D']
    coefs = [1.1525, -2.0303, 0.5769, 0.2120, -0.0010, -4.0845]
    ci_lower = [0.671, -3.232, -0.381, -4.101, -0.002, -87.177]
    ci_upper = [1.634, -0.829, 1.535, 4.525, 0.000, 79.008]
    p_values = [0.000, 0.001, 0.238, 0.923, 0.078, 0.923]

    # Exclude wide R&D CI from main visual scale so other features are clear, or zoom appropriately
    # We display the top predictors clearly
    df_coef = pd.DataFrame({
        'Feature': display_names,
        'Coef': coefs,
        'Lower': ci_lower,
        'Upper': ci_upper,
        'P_value': p_values
    })
    
    # Sort by absolute coefficient for presentation (excluding R&D outlier scale for plot readability)
    plot_df = df_coef[df_coef['Feature'] != 'R&D'].copy()
    plot_df['AbsCoef'] = plot_df['Coef'].abs()
    plot_df = plot_df.sort_values(by='AbsCoef', ascending=True)

    y_pos = np.arange(len(plot_df))
    err_low = plot_df['Coef'] - plot_df['Lower']
    err_high = plot_df['Upper'] - plot_df['Coef']

    colors = ['#2ca02c' if p < 0.05 else '#7f7f7f' for p in plot_df['P_value']]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = '#1b7837' if row['P_value'] < 0.05 else '#636363'
        ax.errorbar(row['Coef'], i, xerr=[[row['Coef'] - row['Lower']], [row['Upper'] - row['Coef']]],
                    fmt='o', color=color, ecolor=color, elinewidth=2, capsize=5, capthick=1.5, markersize=7)
        # Annotation for p-value
        sig_star = "***" if row['P_value'] < 0.001 else ("**" if row['P_value'] < 0.01 else ("*" if row['P_value'] < 0.05 else "ns"))
        ax.annotate(f"β={row['Coef']:.2f} ({sig_star})", xy=(row['Upper'], i),
                    xytext=(6, -3), textcoords='offset points', fontsize=9, fontweight='bold', color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['Feature'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Coefficient Estimate (95% Clustered CI)', fontsize=11, fontweight='bold')
    ax.set_title('OLS Multiple Regression: Feature Effects on AI Rate\n(Clustered Standard Errors by Country)', fontsize=12, fontweight='bold', pad=10)
    ax.text(0.02, 0.05, "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant\n(Note: R&D has wide SE due to collinearity)",
            transform=ax.transAxes, fontsize=8.5, style='italic', bbox=dict(facecolor='#f9f9f9', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    file_path = os.path.join(output_dir, "3_ols_coefficients_forest_plot.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_path}")

# -------------------------------------------------------------
# GRAPH 4: Temporal Trends - Actual vs. Predicted AI Rate (2019-2023)
# -------------------------------------------------------------
def plot_temporal_trends(df):
    _, y_pred = get_loocv_predictions(df)
    
    df = df.copy()
    df['Predicted_AI_Rate'] = y_pred

    # Aggregate mean across all countries per year to show global trend
    trend = df.groupby('Year')[['AI_Rate', 'Predicted_AI_Rate']].mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: Overall Average Trajectory
    ax1.plot(trend['Year'], trend['AI_Rate'], marker='o', linewidth=2.5, color='#1f77b4', label='Actual Mean AI Rate')
    ax1.plot(trend['Year'], trend['Predicted_AI_Rate'], marker='s', linewidth=2.5, linestyle='--', color='#ff7f0e', label='Predicted Mean (LOOCV)')
    ax1.set_title('Aggregate Annual Trend (All Focus Countries)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Mean AI Rate', fontsize=10, fontweight='bold')
    ax1.set_xticks(trend['Year'].astype(int))
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Individual Country Trajectories
    for country in df['Country'].unique():
        c_data = df[df['Country'] == country].sort_values('Year')
        ax2.plot(c_data['Year'], c_data['AI_Rate'], marker='o', alpha=0.7, label=country)

    ax2.set_title('Country-Level AI Rate Volatility & Trajectories', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Actual AI Rate', fontsize=10, fontweight='bold')
    ax2.set_xticks(trend['Year'].astype(int))
    ax2.legend(loc='upper right', frameon=True, fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('Temporal Trends in AI Rate and Model Performance (2019–2023)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    file_path = os.path.join(output_dir, "4_temporal_trends_performance.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_path}")

# -------------------------------------------------------------
# GRAPH 5: SHAP Feature Importance Ranking
# -------------------------------------------------------------
def plot_shap_importance():
    features = ['Investment', 'Broadband', 'R&D', 'Electricity', 'Stability', 'GDP per capita'][::-1]
    shap_vals = [8.544, 4.112, 1.798, 1.141, 0.646, 0.631][::-1]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    bars = ax.barh(features, shap_vals, color='#3182bd', edgecolor='black', linewidth=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f'{w:.2f}', xy=(w, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", ha='left', va='center',
                    fontsize=9.5, fontweight='bold')

    ax.set_xlabel('Mean |SHAP Value| (Average Absolute Impact on AI Rate Prediction)', fontsize=10.5, fontweight='bold')
    ax.set_title('SHAP Feature Importance: Drivers of AI Adoption Rate', fontsize=12.5, fontweight='bold', pad=12)
    ax.set_xlim(0, 10)
    plt.tight_layout()

    file_path = os.path.join(output_dir, "5_shap_feature_importance.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_path}")

if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    print("Generating visualizations...")
    plot_cv_metrics()
    plot_actual_vs_predicted(df)
    plot_ols_coefficients()
    plot_temporal_trends(df)
    plot_shap_importance()
    print(f"\nAll visualizations successfully generated in: {output_dir}")
