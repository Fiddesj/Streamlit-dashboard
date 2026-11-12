import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn imports for ML and Validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Statsmodels for Linear Regression with Clustered Standard Errors
import statsmodels.api as sm

# SHAP for Model Explainability
import shap

# 1. DEFINE VARIABLES & PREPARE DATA
features = [
    'GDP_per_capita', 
    'Broadband', 
    'Electricity', 
    'Investment', 
    'R_and_D', 
    'Stability'
]
target = 'AI_Rate' # target variable
group_col = 'Country_label' # Used for clustering and Leave-One-Country-Out

def load_and_preprocess_data(excel_path=r"C:\Users\Junior\Downloads\AI_Research_Data_Master (1) (1).xlsx"):
    """
    Loads and integrates data across the sheets of AI_Research_Data_Master:
    - Political_Analysis -> Stability (Stability_Index)
    - Digital Infrastructure Score -> Electricity, Broadband
    - 1_Predictor_Data -> GDP_per_capita, R_and_D (converted to numeric)
    - 3_ADI_Calculation -> Investment, AI_Rate (target)
    """
    # 1. Political Analysis (Stability)
    df_pol = pd.read_excel(excel_path, sheet_name='Political_Analysis')
    df_pol['Country'] = df_pol['Country'].astype(str).str.strip().replace({'United States': 'USA'})
    df_pol['Year'] = pd.to_numeric(df_pol['Year'], errors='coerce')
    pol_sub = df_pol[['Country', 'Year', 'Stability_Index']].rename(columns={'Stability_Index': 'Stability'})

    # 2. Digital Infrastructure (Electricity, Broadband)
    df_dig = pd.read_excel(excel_path, sheet_name='Digital Infrastructure Score')
    df_dig['Country'] = df_dig['Country'].astype(str).str.strip().replace({'United States': 'USA'})
    df_dig['Year'] = pd.to_numeric(df_dig['Year'], errors='coerce')
    elec_col = [c for c in df_dig.columns if 'elec' in str(c).lower()][0]
    broad_col = [c for c in df_dig.columns if 'broadband' in str(c).lower()][0]
    dig_sub = df_dig[['Country', 'Year', elec_col, broad_col]].rename(columns={
        elec_col: 'Electricity',
        broad_col: 'Broadband'
    })

    # 3. Predictor Data (GDP per capita, R&D)
    df_pred = pd.read_excel(excel_path, sheet_name='1_Predictor_Data', header=3)
    df_pred['Country'] = df_pred['Country'].astype(str).str.strip().replace({'United States': 'USA'})
    df_pred['Year'] = pd.to_numeric(df_pred['Year'], errors='coerce')
    gdp_col = [c for c in df_pred.columns if 'gdp per capita' in str(c).lower()][0]
    rd_col = [c for c in df_pred.columns if 'r&d' in str(c).lower()][0]
    pred_sub = df_pred[['Country', 'Year', gdp_col, rd_col]].rename(columns={
        gdp_col: 'GDP_per_capita',
        rd_col: 'R_and_D'
    })

    # 4. ADI Calculation (Investment, AI_Rate)
    df_adi = pd.read_excel(excel_path, sheet_name='3_ADI_Calculation', header=3)
    df_adi['Country'] = df_adi['Country'].astype(str).str.strip().replace({'United States': 'USA'})
    df_adi['Year'] = pd.to_numeric(df_adi['Year'], errors='coerce')
    inv_col = [c for c in df_adi.columns if 'investment' in str(c).lower() and 'sub-index' in str(c).lower()][0]
    target_col = [c for c in df_adi.columns if 'rate' in str(c).lower() or 'adr' in str(c).lower()][0]
    adi_sub = df_adi[['Country', 'Year', inv_col, target_col]].rename(columns={
        inv_col: 'Investment',
        target_col: 'AI_Rate'
    })

    # Merge all sheets
    df = pred_sub.merge(dig_sub, on=['Country', 'Year'], how='inner')
    df = df.merge(pol_sub, on=['Country', 'Year'], how='inner')
    df = df.merge(adi_sub, on=['Country', 'Year'], how='inner')

    df['Country_label'] = df['Country']

    # Convert all numeric features and target to numeric
    numeric_cols = ['GDP_per_capita', 'Broadband', 'Electricity', 'Investment', 'R_and_D', 'Stability', 'AI_Rate']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Forward-fill R_and_D per country for recent lag years 
    df['R_and_D'] = df.groupby('Country')['R_and_D'].ffill()

    # Drop base year (2018) where AI_Rate is undefined/NaN
    df = df.dropna(subset=features + [target, group_col]).reset_index(drop=True)

    print(f"Loaded dataset successfully: {len(df)} observations across {df['Country_label'].nunique()} countries.")
    return df

# 2. MULTIPLE LINEAR REGRESSION (Statsmodels)
def run_linear_regression(X, y, groups):
    """
    Runs Multiple Linear Regression applying clustered standard errors 
    at the country level as specified in the methodology
    """
    print("--- Multiple Linear Regression Results ---")
    
    # Add a constant (beta_0 intercept) to the features
    X_with_const = sm.add_constant(X)
    
    # Fit the OLS model
    model = sm.OLS(y, X_with_const)
    
    # clustered standard errors based on the country
    results = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
    
    # Print summary which includes R-squared, Adjusted R-squared, and p-values
    print(results.summary())
    
    return results

# 3. RANDOM FOREST REGRESSION & VALIDATION
def run_random_forest_loocv(X, y, groups):
    """
    Trains a Random Forest using Leave-One-Country-Out cross validation,
    with Grid Search for conservative hyperparameters
    """
    print("\n--- Random Forest (Leave-One-Country-Out CV) ---")
    
    # Define hyperparameter grid prioritizing shallower trees
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [2, 3, 4], # Kept shallow to prevent overfitting
        'min_samples_leaf': [2, 4] # Higher minimum samples
    }
    
    rf = RandomForestRegressor(random_state=42)
    logo = LeaveOneGroupOut()
    
    # GridSearchCV wrapped inside LOOCV loop
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, # Internal CV for tuning
        scoring='neg_mean_absolute_error'
    )
    
    y_true_all, y_pred_all = [], []
    
    # Leave-One-Country-Out Split
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit model on 5 countries, test on the 1 left out
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        preds = best_model.predict(X_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        
    # Calculate overall performance metrics
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2 = r2_score(y_true_all, y_pred_all)
    
    print(f"LOOCV Overall MAE: {mae:.2f}")
    print(f"LOOCV Overall RMSE: {rmse:.2f}")
    print(f"LOOCV Overall R-squared: {r2:.2f}")
    
    # Fit a final model on all data for SHAP extraction
    final_model = grid_search.fit(X, y).best_estimator_
    
    return final_model

# 4. 5-FOLD CROSS-VALIDATION (Supplementary Validation)
def run_5fold_cv(X, y, best_params=None):
    """
    Performs standard 5-Fold Cross-Validation (5 iterations with an 80% training 
    and 20% validation split) as a supplementary method to assess generalization 
    performance and guard against overfitting.
    """
    print("\n--- 5-Fold Cross-Validation (80% Train / 20% Val Split) ---")
    
    # Use tuned parameters if provided, else reasonable defaults
    if best_params:
        params = best_params.copy()
        params['random_state'] = 42
        rf = RandomForestRegressor(**params)
    else:
        rf = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42)
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        rf.fit(X_train, y_train)
        preds = rf.predict(X_val)
        
        fold_mae = mean_absolute_error(y_val, preds)
        fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_r2 = r2_score(y_val, preds)
        
        mae_scores.append(fold_mae)
        rmse_scores.append(fold_rmse)
        r2_scores.append(fold_r2)
        
        print(f"  Fold {fold} - MAE: {fold_mae:.2f}, RMSE: {fold_rmse:.2f}, R²: {fold_r2:.2f}")
        
    print(f"\n5-Fold Mean MAE:  {np.mean(mae_scores):.2f} (± {np.std(mae_scores):.2f})")
    print(f"5-Fold Mean RMSE: {np.mean(rmse_scores):.2f} (± {np.std(rmse_scores):.2f})")
    print(f"5-Fold Mean R²:   {np.mean(r2_scores):.2f} (± {np.std(r2_scores):.2f})")
    
    return {
        'mae_mean': np.mean(mae_scores),
        'rmse_mean': np.mean(rmse_scores),
        'r2_mean': np.mean(r2_scores)
    }

# 5. SHAP EXPLAINABILITY ANALYSIS
def run_shap_analysis(model, X):
    """
    Extracts SHAP values to rank feature importance and explain 
    individual country-year predictions[cite: 26].
    """
    print("\n--- Generating SHAP Analysis ---")
    
    # Initialize the Tree Explainer for Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 1. SHAP Summary Plot: Shows overall feature importance and direction
    plt.figure()
    plt.title("SHAP Summary Plot")
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("SHAP Summary Plot saved as 'shap_summary_plot.png'")
    
    # 2. Mean Absolute SHAP values (Feature Ranking)
    # shap_values is a numpy array; we take the absolute mean across all 42 observations
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_df = pd.DataFrame(list(zip(X.columns, mean_abs_shap)), columns=['Feature', 'Mean_Absolute_SHAP'])
    shap_df = shap_df.sort_values(by='Mean_Absolute_SHAP', ascending=False)
    print("\nFeature Importance (Mean Absolute SHAP):")
    print(shap_df)
    
    # 3. Individual Observation Force Plot 
    print("\nGenerating Force Plot for observation 0...")
    try:
        force_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
        shap.save_html("shap_force_plot_obs0.html", force_plot)
        print("Force plot saved to 'shap_force_plot_obs0.html'")
    except Exception as e:
        print(f"Force plot skipped in standalone script mode ({e}). Use Jupyter notebook for interactive display.")

# 6. ROBUSTNESS AND EDGE-CASE ANALYSIS
def run_robustness_and_edge_cases(rf_model, X):
    """
    Tests model stability under feature perturbations (shocks) and 
    evaluates model behavior on extreme synthetic edge cases.
    """
    print("\n--- Robustness & Edge Case Analysis ---")

    # Baseline predictions on untouched data
    base_predictions = rf_model.predict(X)
    base_mean = np.mean(base_predictions)

    # 1. Perturbation (Shock Test)
    print("\n1. Sensitivity to Predictor Shocks (Mean Predicted AI Rate):")
    print(f"   Baseline Mean Prediction: {base_mean:.2f}")

    for col in ["Investment", "Broadband", "R_and_D"]:
        X_hike = X.copy()
        X_drop = X.copy()

        X_hike[col] = X_hike[col] * 1.20  # 20% hike
        X_drop[col] = X_drop[col] * 0.80  # 20% drop

        hike_diff = np.mean(rf_model.predict(X_hike)) - base_mean  
        drop_diff = np.mean(rf_model.predict(X_drop)) - base_mean

        print(f"   {col:12s} | +20% Hike: {hike_diff:+.2f} change | -20% Drop: {drop_diff:+.2f} change")  

    # 2. Synthetic Edge-Case Scenarios
    print("\n2. Extreme Edge-Case Scenarios:")
    edge_cases = pd.DataFrame([
        # Case A: Investment Freeze (Investment = 0, everything else average)
        {**X.mean().to_dict(), "Investment": 0.0},
        # Case B: Maximum Infrastructure & Investment Boom (all features at max observed)
        X.max().to_dict(),
        # Case C: Severe Economic/Infrastructure Collapse (all features at min observed)
        X.min().to_dict(),
        # Case D: Extreme Out-of-Bounds Broadband Hike (10x normal Broadband)
        {**X.mean().to_dict(), "Broadband": X["Broadband"].max() * 10}
    ], index=[
        "Zero Investment Freeze",
        "All-Maximum Boom",
        "All-Minimum Collapse",
        "Extreme Broadband Outlier (10x Max)"
    ])
    
    edge_predictions = rf_model.predict(edge_cases[X.columns])
    for scenario, pred in zip(edge_cases.index, edge_predictions):
        print(f"   {scenario:36s} -> Predicted AI_Rate: {pred:.2f}")

# EXECUTION WORKFLOW
if __name__ == '__main__':
    # 1. Load data from AI_Research_Data_Master (1).xlsx
    df = load_and_preprocess_data()
    
    # 2. Extract features, target, and groups
    X = df[features]
    y = df[target]
    groups = df[group_col]
    
    print("\nDataset Preview (First 5 rows):")
    print(df[['Country_label', 'Year'] + features + [target]].head())
    
    # 3. Multiple Linear Regression with Clustered Standard Errors
    linear_results = run_linear_regression(X, y, groups)
    
    # 4. Random Forest with Leave-One-Country-Out Cross-Validation
    rf_model = run_random_forest_loocv(X, y, groups)
    
    # 5. 5-Fold Cross-Validation (Supplementary Validation)
    cv_5fold_results = run_5fold_cv(X, y, best_params=rf_model.get_params())
    
    # 6. SHAP Analysis
    run_shap_analysis(rf_model, X) 

    # 7. Robustness & Edge Case Analysis
    run_robustness_and_edge_cases(rf_model, X)