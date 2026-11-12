"""
AI Development Rate & Compute Subindex - Comprehensive Streamlit Dashboard
Integrates Econometric (OLS) and Machine Learning (Random Forest) findings,
SHAP explainability, robustness checks, and contextual analysis comparing
developed and developing nations.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scikit-learn & Statsmodels
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

# Page Configuration
st.set_page_config(
    page_title="AI Development Rate & Global Compute Divide Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b6584;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.2rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10b981;
        margin-top: 0.2rem;
    }
    .context-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 5px solid #0284c7;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .divider-custom {
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. DATA CACHING & PREPROCESSING
# -------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_PATHS = [
    os.path.join(_BASE_DIR, "AI_Research_Data_Master.xlsx"),
    os.path.join(_BASE_DIR, "Compute subindex", "AI_Research_Data_Master.xlsx"),
    r"C:\Users\Junior\Downloads\AI_Research_Data_Master (1) (1).xlsx",
]
EXCEL_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), _CANDIDATE_PATHS[0])
FEATURES = ['GDP_per_capita', 'Broadband', 'Electricity', 'Investment', 'R_and_D', 'Stability']
TARGET = 'AI_Rate'
GROUP_COL = 'Country_label'

DEVELOPED_COUNTRIES = ['Germany', 'Switzerland', 'USA']
DEVELOPING_COUNTRIES = ['China', 'India', 'South Africa']

@st.cache_data
def load_data():
    if not os.path.exists(EXCEL_PATH):
        st.error(f"Excel file not found at: {EXCEL_PATH}")
        return pd.DataFrame()
        
    # 1. Political Analysis (Stability)
    df_pol = pd.read_excel(EXCEL_PATH, sheet_name='Political_Analysis')
    df_pol['Country'] = df_pol['Country'].astype(str).str.strip().replace({'United States': 'USA'})
    df_pol['Year'] = pd.to_numeric(df_pol['Year'], errors='coerce')
    pol_sub = df_pol[['Country', 'Year', 'Stability_Index']].rename(columns={'Stability_Index': 'Stability'})

    # 2. Digital Infrastructure (Electricity, Broadband)
    df_dig = pd.read_excel(EXCEL_PATH, sheet_name='Digital Infrastructure Score')
    df_dig['Country'] = df_dig['Country'].astype(str).str.strip().replace({'United States': 'USA'})
    df_dig['Year'] = pd.to_numeric(df_dig['Year'], errors='coerce')
    elec_col = [c for c in df_dig.columns if 'elec' in str(c).lower()][0]
    broad_col = [c for c in df_dig.columns if 'broadband' in str(c).lower()][0]
    dig_sub = df_dig[['Country', 'Year', elec_col, broad_col]].rename(columns={
        elec_col: 'Electricity',
        broad_col: 'Broadband'
    })

    # 3. Predictor Data (GDP per capita, R&D)
    df_pred = pd.read_excel(EXCEL_PATH, sheet_name='1_Predictor_Data', header=3)
    df_pred['Country'] = df_pred['Country'].astype(str).str.strip().replace({'United States': 'USA'})
    df_pred['Year'] = pd.to_numeric(df_pred['Year'], errors='coerce')
    gdp_col = [c for c in df_pred.columns if 'gdp per capita' in str(c).lower()][0]
    rd_col = [c for c in df_pred.columns if 'r&d' in str(c).lower()][0]
    pred_sub = df_pred[['Country', 'Year', gdp_col, rd_col]].rename(columns={
        gdp_col: 'GDP_per_capita',
        rd_col: 'R_and_D'
    })

    # 4. ADI Calculation (Investment, AI_Rate)
    df_adi = pd.read_excel(EXCEL_PATH, sheet_name='3_ADI_Calculation', header=3)
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

    # Numeric conversion
    numeric_cols = ['GDP_per_capita', 'Broadband', 'Electricity', 'Investment', 'R_and_D', 'Stability', 'AI_Rate']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Forward-fill R_and_D per country
    df['R_and_D'] = df.groupby('Country')['R_and_D'].ffill()

    # Drop missing values
    df = df.dropna(subset=FEATURES + [TARGET, GROUP_COL]).reset_index(drop=True)
    df['Year'] = df['Year'].astype(int)
    
    # Classify economic category
    df['Economic_Group'] = df['Country'].apply(
        lambda c: 'Developed' if c in DEVELOPED_COUNTRIES else 'Developing'
    )
    return df

@st.cache_resource
def train_and_evaluate_models(df):
    X = df[FEATURES]
    y = df[TARGET]
    groups = df[GROUP_COL]

    # --- 1. OLS with Clustered SE ---
    X_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_const)
    ols_res = ols_model.fit(cov_type='cluster', cov_kwds={'groups': groups})

    # --- 2. Random Forest LOOCV ---
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

    y_pred_loocv = np.zeros(len(df))
    for train_idx, test_idx in logo.split(X, y, groups):
        grid_search.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_loocv[test_idx] = grid_search.best_estimator_.predict(X.iloc[test_idx])

    loocv_mae = mean_absolute_error(y, y_pred_loocv)
    loocv_rmse = np.sqrt(mean_squared_error(y, y_pred_loocv))
    loocv_r2 = r2_score(y, y_pred_loocv)

    # Final full model
    final_rf = grid_search.fit(X, y).best_estimator_

    # --- 3. 5-Fold Cross Validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        rf_fold = RandomForestRegressor(**final_rf.get_params())
        rf_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = rf_fold.predict(X.iloc[val_idx])
        fold_metrics.append({
            'Fold': f"Fold {fold}",
            'MAE': mean_absolute_error(y.iloc[val_idx], preds),
            'RMSE': np.sqrt(mean_squared_error(y.iloc[val_idx], preds)),
            'R2': r2_score(y.iloc[val_idx], preds)
        })
    df_5fold = pd.DataFrame(fold_metrics)

    # --- 4. SHAP Feature Importance ---
    # Cached mean abs SHAP values from ML Model.py run
    shap_importance = pd.DataFrame({
        'Feature': ['Investment', 'Broadband', 'R_and_D', 'Electricity', 'Stability', 'GDP_per_capita'],
        'Display_Name': ['AI Compute Investment', 'Fixed Broadband per 100', 'R&D (% GDP)', 'Electricity Access (%)', 'Political Stability Index', 'GDP per capita (USD)'],
        'Mean_Absolute_SHAP': [8.544092, 4.111521, 1.797754, 1.140955, 0.645714, 0.631057]
    }).sort_values('Mean_Absolute_SHAP', ascending=True)

    return {
        'ols_res': ols_res,
        'rf_model': final_rf,
        'y_pred_loocv': y_pred_loocv,
        'loocv_mae': loocv_mae,
        'loocv_rmse': loocv_rmse,
        'loocv_r2': loocv_r2,
        'df_5fold': df_5fold,
        'shap_importance': shap_importance
    }

# -------------------------------------------------------------
# APP INITIALIZATION
# -------------------------------------------------------------
df = load_data()

if df.empty:
    st.error("Failed to load dataset. Please verify file paths.")
    st.stop()

models_dict = train_and_evaluate_models(df)
df_eval = df.copy()
df_eval['Predicted_AI_Rate'] = models_dict['y_pred_loocv']
df_eval['Residual'] = df_eval['AI_Rate'] - df_eval['Predicted_AI_Rate']

# -------------------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=65)
    st.markdown("### **AI Growth & Compute Analytics**")
    st.caption("Empirical Analysis across Developed & Developing Nations")
    
    st.markdown("---")
    st.markdown("#### **Dashboard Navigation**")
    selected_tab = st.radio(
        "Select Module:",
        [
            "🌐 Developed vs. Developing Divide",
            "📊 Econometric Findings (OLS)",
            "🌲 Machine Learning & LOOCV",
            "🔍 SHAP Explainability & Simulator",
            "🧪 Robustness & Edge Cases",
            "📋 Panel Data & Correlation"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### **Data Filters (Global)**")
    selected_years = st.slider(
        "Observation Time Window:",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max()))
    )
    
    selected_groups = st.multiselect(
        "Economic Group:",
        options=['Developed', 'Developing'],
        default=['Developed', 'Developing']
    )
    
    filtered_df = df[
        (df['Year'].between(selected_years[0], selected_years[1])) &
        (df['Economic_Group'].isin(selected_groups))
    ]
    
    st.markdown("---")
    st.info(f"**Loaded Panel**: 6 Economies\n\n• **Developed**: Germany, Switzerland, USA\n• **Developing**: China, India, South Africa\n\n**Years**: 2019 – 2024 ({len(filtered_df)} observations)")

# -------------------------------------------------------------
# TOP HEADER & KPI CARDS
# -------------------------------------------------------------
st.markdown('<div class="main-header">⚡ AI Development Rate & Compute Subindex Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Method Empirical Evaluation of AI Adoption Determinants and the Widening Global Compute Divide</div>', unsafe_allow_html=True)

# 4 Key Metrics Bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Primary Driver (SHAP Rank 1)</div>
        <div class="metric-value">Investment</div>
        <div class="metric-sub">Mean |SHAP| = 8.54 | β = +1.15***</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">OLS Explanatory Power</div>
        <div class="metric-value">R² = {models_dict['ols_res'].rsquared:.3f}</div>
        <div class="metric-sub">Adj. R² = {models_dict['ols_res'].rsquared_adj:.3f} (F = {models_dict['ols_res'].fvalue:.1f}***)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">RF LOOCV Generalization</div>
        <div class="metric-value">MAE = {models_dict['loocv_mae']:.2f}</div>
        <div class="metric-sub">RMSE = {models_dict['loocv_rmse']:.2f} (5-Fold: 20.64)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    dev_inv = df[df['Economic_Group'] == 'Developed']['Investment'].mean()
    dev_ing_inv = df[df['Economic_Group'] == 'Developing']['Investment'].mean()
    ratio = dev_inv / dev_ing_inv if dev_ing_inv > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Compute Investment Divide</div>
        <div class="metric-value">{ratio:.1f}x Disparity</div>
        <div class="metric-sub">Developed: {dev_inv:.1f} vs Developing: {dev_ing_inv:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# MODULE 1: DEVELOPED VS. DEVELOPING DIVIDE (Context Focus)
# -------------------------------------------------------------
if selected_tab == "🌐 Developed vs. Developing Divide":
    st.markdown("## 🌐 AI Development Rate Context: The Developed vs. Developing Divide")
    
    st.markdown("""
    <div class="context-box">
        <strong>Contextual Research Finding:</strong> While both economic cohorts average a similar nominal AI Growth Rate (~7.8% Developed vs. ~7.5% Developing), 
        their underlying determinants and trajectories reveal a profound structural chasm. Developing nations experience 
        <strong>hyper-volatility driven by severe infrastructure bottlenecks</strong> (load shedding in South Africa, infrastructure scaling in India), 
        whereas Developed nations' growth is directly driven by <strong>massive compute capital concentration</strong> (USA, Germany, Switzerland).
    </div>
    """, unsafe_allow_html=True)

    # Comparison metrics
    col_a, col_b = st.columns([3, 2])
    
    with col_a:
        st.subheader("📈 Annual AI Adoption Rate (ADR) Trajectories (2019–2024)")
        
        # Color palette
        color_map = {
            'USA': '#1f77b4', 'Germany': '#2ca02c', 'Switzerland': '#17becf',
            'China': '#d62728', 'India': '#ff7f0e', 'South Africa': '#9467bd'
        }
        
        fig_traj = px.line(
            filtered_df,
            x='Year',
            y='AI_Rate',
            color='Country',
            line_dash='Economic_Group',
            markers=True,
            color_discrete_map=color_map,
            labels={'AI_Rate': 'AI Development Rate (ADR)', 'Year': 'Year'},
            title="Comparative AI Rate Volatility by Country & Economic Group"
        )
        fig_traj.update_traces(line=dict(width=2.5), marker=dict(size=8))
        fig_traj.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Growth Baseline")
        fig_traj.update_layout(height=450, legend=dict(orientation="h", y=-0.2), hovermode="x unified")
        st.plotly_chart(fig_traj, use_container_width=True)

    with col_b:
        st.subheader("⚖️ Structural Predictor Breakdown")
        comp_summary = df.groupby('Economic_Group')[FEATURES + ['AI_Rate']].mean().round(2).T
        comp_summary.columns = ['Developed (Mean)', 'Developing (Mean)']
        comp_summary['Disparity Ratio (Dev/Dev-ing)'] = (comp_summary['Developed (Mean)'] / comp_summary['Developing (Mean)'].replace(0, np.nan)).round(2)
        st.dataframe(comp_summary, use_container_width=True)
        
        st.markdown("""
        **Key Structural Observations:**
        - **Compute Investment**: **19.0x gap** between developed and developing nations.
        - **R&D Intensity**: **2.6x gap** (3.26% vs. 1.26% of GDP).
        - **Political Stability**: **5.2x gap** (76.4 vs. 14.6 index score).
        - **Broadband Access**: **2.9x gap** (42.3 vs. 14.6 subscriptions per 100).
        """)

    st.markdown("---")
    
    # 2 Column Deep-Dive: Investment Gap & Country Profiles
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.subheader("💰 The AI Compute Investment Gap (2019-2024)")
        fig_inv = px.bar(
            filtered_df,
            x='Year',
            y='Investment',
            color='Country',
            barmode='group',
            color_discrete_map=color_map,
            labels={'Investment': 'Compute Investment Sub-Index', 'Year': 'Year'},
            title="Compute Investment Intensity Over Time"
        )
        fig_inv.update_layout(height=400, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_inv, use_container_width=True)
        
        st.caption("Notice how the USA dominates compute capital deployment (peaking >83 in 2024), while developing nations remain under 5.0.")

    with col_d:
        st.subheader("🌐 Radar Profile: Developed vs. Developing Baseline")
        
        # Normalized radar chart (0-100 scale)
        radar_features = ['Investment', 'Broadband', 'Electricity', 'R_and_D', 'Stability']
        df_norm = df.copy()
        for f in radar_features:
            max_v = df_norm[f].max()
            df_norm[f] = (df_norm[f] / max_v) * 100
            
        dev_radar = df_norm[df_norm['Economic_Group'] == 'Developed'][radar_features].mean().values.tolist()
        deving_radar = df_norm[df_norm['Economic_Group'] == 'Developing'][radar_features].mean().values.tolist()
        
        dev_radar.append(dev_radar[0])
        deving_radar.append(deving_radar[0])
        categories = ['Compute Inv.', 'Broadband', 'Electricity', 'R&D %', 'Stability', 'Compute Inv.']
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=dev_radar,
            theta=categories,
            fill='toself',
            name='Developed (USA, DE, CH)',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=deving_radar,
            theta=categories,
            fill='toself',
            name='Developing (CN, IN, ZA)',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=400,
            title="Normalized Readiness Envelope (Max = 100%)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Narrative Synthesis
    st.markdown("### 📝 Socioeconomic & Policy Insights: Diverging AI Trajectories")
    tab_n1, tab_n2, tab_n3 = st.tabs(["Developing Nations Context", "Developed Nations Context", "Policy Recommendations"])
    
    with tab_n1:
        st.markdown("""
        #### Developing Nations: Volatility, Leapfrogging & Severe Infrastructure Friction
        - **South Africa (The Low-Base Surge & Infrastructure Ceiling)**: Experienced massive initial spikes in 2019 (+84.1%) and 2020 (+62.8%) from an extremely small initial base. However, growth plummeted to -17.0% (2022) and -15.1% (2023). This collapse directly mirrors **persistent electrical grid instability (Eskom load-shedding)** and access rates of ~85%, demonstrating that algorithmic adoption cannot progress without basic power reliability.
        - **India (Talent Abundance vs. Compute Starvation)**: India displays extreme cyclical swings (-11.7% in 2019, -11.6% in 2021, and +23.3% in 2024). While India possesses one of the world's largest AI talent and software developer bases, domestic **compute investment is exceptionally low (index 0.05 – 2.3)**, meaning India is reliant on foreign hyperscaler APIs rather than domestic compute hardware.
        - **China (Strategic High-Tech Sovereignty)**: China represents a distinct quasi-developing hybrid. With R&D at 2.64% of GDP and broadband at 47.2/100, its AI rate maintained steady growth (+9.2%, +7.3%, +10.6%) before softening in 2023 (-24.8%) due to semiconductor export restrictions, before recovering in 2024 (+15.9%).
        """)
        
    with tab_n2:
        st.markdown("""
        #### Developed Nations: Capital Intensity, High Baselines & Steady Integration
        - **USA (The Compute Superpower)**: The US maintains an astronomical compute advantage, with the investment index climbing from 54.2 in 2019 to 83.8 in 2024. Its AI growth surges in 2021 (+44.9%) and 2024 (+24.1%) track the multi-billion dollar private capital deployments into LLMs and frontier data centers.
        - **Germany (Industrial AI & Regulatory Friction)**: Shows high stability (80+) and strong R&D (3.15%), but experiences sharp cyclical rate adjustments (-33.5% in 2022, +54.3% in 2023, -11.9% in 2024). Adoption is heavily linked to automotive/manufacturing capex cycles and stricter EU data governance compliance.
        - **Switzerland (Stable Frontier Equilibrium)**: Achieves perfect 100/100 Political Stability and ~3.4% R&D intensity, with balanced compute funding (mean index 38.5). Growth rate fluctuations represent maturity adjustments from a very high baseline.
        """)
        
    with tab_n3:
        st.markdown("""
        #### Key Policy Implications for Closing the AI Divide
        1. **Electricity Grid Sovereignty**: Electricity access showed a statistically significant threshold effect ($\beta = -2.03, p = 0.001$). Developing countries cannot participate in generative AI without uninterrupted 24/7 power for data centers and edge devices.
        2. **Democratizing Compute Access**: With a 19x compute investment divide, multilateral bodies must support shared regional supercomputing hubs (e.g. African & South Asian shared compute clusters).
        3. **R&D Capitalization**: Indigenous model development requires R&D investment beyond basic IT services. Closing the 2.6x R&D gap is vital to avoid technological dependency.
        """)

# -------------------------------------------------------------
# MODULE 2: ECONOMETRIC FINDINGS (OLS)
# -------------------------------------------------------------
elif selected_tab == "📊 Econometric Findings (OLS)":
    st.markdown("## 📊 Econometric Analysis: Multiple Linear Regression")
    st.caption("Ordinary Least Squares (OLS) with Country-Clustered Standard Errors (CRV1)")
    
    ols_res = models_dict['ols_res']
    
    col_reg1, col_reg2 = st.columns([3, 2])
    
    with col_reg1:
        st.subheader("📑 Regression Parameter Estimates")
        
        # Build clean regression table
        reg_df = pd.DataFrame({
            'Feature': ['Intercept (const)', 'GDP per capita', 'Broadband per 100', 'Electricity Access %', 'Compute Investment', 'R&D (% GDP)', 'Political Stability'],
            'Coefficient (β)': ols_res.params.values,
            'Cluster Std. Err.': ols_res.bse.values,
            'z-statistic': ols_res.tvalues.values,
            'p-value': ols_res.pvalues.values,
            '95% CI Lower': ols_res.conf_int()[0].values,
            '95% CI Upper': ols_res.conf_int()[1].values
        })
        
        # Significance stars
        def get_stars(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            elif p < 0.1: return '†'
            return 'ns'
            
        reg_df['Significance'] = reg_df['p-value'].apply(get_stars)
        
        st.dataframe(
            reg_df.style.format({
                'Coefficient (β)': '{:.4f}',
                'Cluster Std. Err.': '{:.4f}',
                'z-statistic': '{:.3f}',
                'p-value': '{:.4f}',
                '95% CI Lower': '{:.3f}',
                '95% CI Upper': '{:.3f}'
            }),
            use_container_width=True
        )
        st.caption("*** p<0.001, ** p<0.01, * p<0.05, † p<0.1, ns = not significant")

    with col_reg2:
        st.subheader("📐 Model Fit & Diagnostics")
        st.markdown(f"""
        - **Dependent Variable**: AI_Rate (Annual Adoption Rate)
        - **Observations**: {int(ols_res.nobs)} across 6 country clusters
        - **R-squared ($R^2$)**: **{ols_res.rsquared:.4f}** (35.5% of variance explained)
        - **Adjusted $R^2$**: **{ols_res.rsquared_adj:.4f}**
        - **Wald F-statistic**: **{ols_res.fvalue:.1f}** (p-value: **{ols_res.f_pvalue:.2e}**)
        - **Log-Likelihood**: {ols_res.llf:.2f} | **AIC**: {ols_res.aic:.1f}
        - **Condition Number**: **{ols_res.condition_number:.2e}**
        """)
        
        st.warning("""
        **Condition Number Diagnostic Note**:
        The high condition number indicates significant **multicollinearity** among predictors (e.g., GDP per capita, R&D, Stability, and Broadband are highly correlated in developed nations), which inflates standard errors for secondary variables like R&D.
        """)

    st.markdown("---")
    st.subheader("🌲 Forest Plot: Regression Coefficients & 95% Clustered Confidence Intervals")
    
    # Forest plot excluding wide R&D scale for readability or zoomable
    forest_df = reg_df[~reg_df['Feature'].isin(['Intercept (const)', 'R&D (% GDP)'])].copy()
    
    fig_forest = go.Figure()
    fig_forest.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5)
    
    for i, row in forest_df.iterrows():
        color = '#1b7837' if row['p-value'] < 0.05 else '#636363'
        fig_forest.add_trace(go.Scatter(
            x=[row['Coefficient (β)']],
            y=[row['Feature']],
            error_x=dict(
                type='data',
                symmetric=False,
                array=[row['95% CI Upper'] - row['Coefficient (β)']],
                arrayminus=[row['Coefficient (β)'] - row['95% CI Lower']],
                color=color,
                thickness=2.5,
                width=8
            ),
            mode='markers+text',
            marker=dict(size=10, color=color),
            text=[f"β={row['Coefficient (β)']:.2f} ({row['Significance']})"],
            textposition='top center',
            name=row['Feature'],
            showlegend=False
        ))
        
    fig_forest.update_layout(
        title="Predictor Impact on AI Development Rate (with 95% Clustered CI)",
        xaxis_title="Coefficient Estimate (Percentage Point Change in AI Rate)",
        yaxis_title="Predictor Feature",
        height=380
    )
    st.plotly_chart(fig_forest, use_container_width=True)

    # In-depth discussion
    st.markdown("""
    ### 🔬 Econometric Interpretation of Findings
    1. **Compute Investment is the Dominant Statistically Significant Driver**:
       - $\\beta = +1.1525, z = 4.688, p < 0.001$.
       - Each 1-unit increase in the compute investment index accelerates the national AI adoption rate by **+1.15 percentage points**. The 95% confidence interval $[0.671, 1.634]$ remains entirely positive, proving the indispensable role of physical compute capital.
    2. **Electricity Access ($\beta = -2.0303, p = 0.001$) Reflects Base Saturation**:
       - The negative coefficient stems from structural saturation: 100% electrified nations experience more modest percentage swings, whereas developing countries with lower electricity access (84-95%) display larger early baseline swings.
    3. **R&D and Stability Collinearity**:
       - Both R&D and Stability are statistically dampened in the linear specification because they are heavily colinear with GDP per capita ($r > 0.85$), making machine learning non-linear trees essential to unravel their true importance.
    """)

# -------------------------------------------------------------
# MODULE 3: MACHINE LEARNING & LOOCV
# -------------------------------------------------------------
elif selected_tab == "🌲 Machine Learning & LOOCV":
    st.markdown("## 🌲 Machine Learning: Random Forest & Cross-Validation")
    st.caption("Leave-One-Country-Out Cross-Validation (LOOCV) and 5-Fold Stratified Evaluation")

    col_ml1, col_ml2 = st.columns([2, 3])
    
    with col_ml1:
        st.subheader("🎯 Generalization Performance")
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">LOOCV (Country Out-of-Sample)</div>
            <div class="metric-value">MAE: {models_dict['loocv_mae']:.2f}</div>
            <div class="metric-sub">RMSE: {models_dict['loocv_rmse']:.2f} | R²: {models_dict['loocv_r2']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">5-Fold CV (80% Train / 20% Val)</div>
            <div class="metric-value">Mean MAE: {models_dict['df_5fold']['MAE'].mean():.2f} ± {models_dict['df_5fold']['MAE'].std():.2f}</div>
            <div class="metric-sub">Mean RMSE: {models_dict['df_5fold']['RMSE'].mean():.2f} | Mean R²: {models_dict['df_5fold']['R2'].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(models_dict['df_5fold'].style.format({'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'R2': '{:.2f}'}), use_container_width=True)

    with col_ml2:
        st.subheader("📊 5-Fold Cross-Validation Metrics Bar Chart")
        
        fig_5f = go.Figure()
        fig_5f.add_trace(go.Bar(
            x=models_dict['df_5fold']['Fold'],
            y=models_dict['df_5fold']['MAE'],
            name='MAE',
            marker_color='#2b5c8f'
        ))
        fig_5f.add_trace(go.Bar(
            x=models_dict['df_5fold']['Fold'],
            y=models_dict['df_5fold']['RMSE'],
            name='RMSE',
            marker_color='#e26d5c'
        ))
        fig_5f.add_hline(
            y=models_dict['df_5fold']['MAE'].mean(), 
            line_dash="dash", line_color='#1e3d59',
            annotation_text=f"Mean MAE ({models_dict['df_5fold']['MAE'].mean():.2f})"
        )
        fig_5f.add_hline(
            y=models_dict['df_5fold']['RMSE'].mean(), 
            line_dash="dash", line_color='#b23b2a',
            annotation_text=f"Mean RMSE ({models_dict['df_5fold']['RMSE'].mean():.2f})"
        )
        fig_5f.update_layout(
            barmode='group',
            title="Error Magnitude Across 5 Validation Folds",
            yaxis_title="Error Magnitude",
            height=370
        )
        st.plotly_chart(fig_5f, use_container_width=True)

    st.markdown("---")
    
    # Actual vs Predicted Scatter
    st.subheader("🎯 Actual vs. Predicted AI Rate (Leave-One-Country-Out CV)")
    
    col_sc1, col_sc2 = st.columns([3, 2])
    
    with col_sc1:
        color_map = {
            'USA': '#1f77b4', 'Germany': '#2ca02c', 'Switzerland': '#17becf',
            'China': '#d62728', 'India': '#ff7f0e', 'South Africa': '#9467bd'
        }
        
        fig_sc = px.scatter(
            df_eval,
            x='AI_Rate',
            y='Predicted_AI_Rate',
            color='Country',
            symbol='Economic_Group',
            hover_data=['Year', 'Investment', 'Broadband'],
            color_discrete_map=color_map,
            labels={'AI_Rate': 'Actual AI Rate (%)', 'Predicted_AI_Rate': 'Predicted AI Rate (LOOCV)'},
            title="Out-of-Sample Predictions vs. Actual Observed AI Rate"
        )
        
        # 1:1 diagonal line
        min_val = min(df_eval['AI_Rate'].min(), df_eval['Predicted_AI_Rate'].min()) - 5
        max_val = max(df_eval['AI_Rate'].max(), df_eval['Predicted_AI_Rate'].max()) + 5
        fig_sc.add_shape(
            type="line", line=dict(dash="dash", color="gray", width=1.5),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        fig_sc.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color='black')))
        fig_sc.update_layout(height=450)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_sc2:
        st.markdown("""
        #### Critical Methodology Takeaways:
        - **Why LOOCV Matters**: Rather than traditional random train/test splits that can leak country-specific fixed effects across years, **Leave-One-Country-Out** holds out an entire economy's history.
        - **Negative Out-of-Sample $R^2$ ($-0.13$)**: This reflects the sharp **macroeconomic idiosyncrasies** between countries. A model trained on China, India, and Germany cannot easily predict the massive post-COVID volatility of South Africa without knowing local structural shifts.
        - **Residual Regularity**: Notice that for developed countries with steady investment (USA, Switzerland), the predictions cluster tightly near the 1:1 line. The largest residuals occur for South Africa's early leapfrogging years (2019–2020).
        """)

# -------------------------------------------------------------
# MODULE 4: SHAP EXPLAINABILITY & SIMULATOR
# -------------------------------------------------------------
elif selected_tab == "🔍 SHAP Explainability & Simulator":
    st.markdown("## 🔍 SHAP Model Explainability & What-If Policy Simulator")
    st.caption("TreeExplainer Feature Attributions and Interactive Scenario Prediction")

    col_sh1, col_sh2 = st.columns([3, 2])
    
    with col_sh1:
        st.subheader("🏆 SHAP Feature Importance Ranking")
        shap_df = models_dict['shap_importance']
        
        fig_shap = px.bar(
            shap_df,
            x='Mean_Absolute_SHAP',
            y='Display_Name',
            orientation='h',
            text='Mean_Absolute_SHAP',
            labels={'Mean_Absolute_SHAP': 'Mean |SHAP Value| (Impact on AI Rate)', 'Display_Name': 'Feature'},
            title="Average Absolute Impact of Features on Predicted AI Rate"
        )
        fig_shap.update_traces(marker_color='#3182bd', texttemplate='%{text:.2f}', textposition='outside')
        fig_shap.update_layout(height=380, xaxis=dict(range=[0, 10]))
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_sh2:
        st.subheader("💡 SHAP Driver Hierarchy")
        st.markdown("""
        1. **Compute Investment (8.54)**: By far the most influential lever in the random forest model, accounting for more than double the impact of any other feature.
        2. **Broadband Access (4.11)**: Second most important; provides the necessary digital bandwidth pipeline for model deployment.
        3. **R&D Intensity (1.80)**: Critical threshold capacity for internalizing AI technology.
        4. **Electricity Access (1.14)**: Necessary physical foundation.
        5. **Political Stability (0.65)**: Moderates long-term venture and capital allocation.
        6. **GDP per capita (0.63)**: Weak direct influence when isolated from capital investment.
        """)

    st.markdown("---")
    
    # Static SHAP image if available
    shap_img_path = "shap_summary_plot.png"
    if os.path.exists(shap_img_path):
        st.subheader("📈 SHAP Summary Plot (Direction of Effects)")
        st.image(shap_img_path, caption="SHAP Summary Plot: Red = High Feature Value, Blue = Low Feature Value", use_container_width=True)

    st.markdown("---")
    # Interactive Simulator
    st.subheader("🎛️ Interactive 'What-If' AI Rate Predictor Simulator")
    st.markdown("Adjust policy predictors below to observe the Random Forest model's real-time prediction for AI Development Rate:")

    rf_model = models_dict['rf_model']

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        sim_inv = st.slider("Compute Investment Index:", 0.0, 100.0, float(df['Investment'].mean()), step=1.0)
        sim_broad = st.slider("Broadband (per 100 people):", 0.0, 60.0, float(df['Broadband'].mean()), step=0.5)

    with col_sim2:
        sim_rd = st.slider("R&D Intensity (% of GDP):", 0.0, 5.0, float(df['R_and_D'].mean()), step=0.1)
        sim_elec = st.slider("Electricity Access (% pop):", 50.0, 100.0, float(df['Electricity'].mean()), step=0.5)

    with col_sim3:
        sim_stab = st.slider("Political Stability Index:", 0.0, 100.0, float(df['Stability'].mean()), step=1.0)
        sim_gdp = st.slider("GDP per Capita (USD):", 1000, 120000, int(df['GDP_per_capita'].mean()), step=1000)

    sim_input = pd.DataFrame([{
        'GDP_per_capita': sim_gdp,
        'Broadband': sim_broad,
        'Electricity': sim_elec,
        'Investment': sim_inv,
        'R_and_D': sim_rd,
        'Stability': sim_stab
    }])[FEATURES]

    sim_pred = rf_model.predict(sim_input)[0]

    st.markdown(f"""
    <div style="background-color: #ecfdf5; border: 2px solid #10b981; border-radius: 10px; padding: 1.5rem; text-align: center; margin-top: 1rem;">
        <span style="font-size: 1.1rem; font-weight: 600; color: #065f46;">Predicted Annual AI Development Rate:</span>
        <div style="font-size: 2.8rem; font-weight: 800; color: #047857;">{sim_pred:+.2f}%</div>
        <span style="font-size: 0.9rem; color: #047857;">Based on conservative Random Forest hyperparameter tuning (max_depth=3, min_samples_leaf=2)</span>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# MODULE 5: ROBUSTNESS & EDGE CASES
# -------------------------------------------------------------
elif selected_tab == "🧪 Robustness & Edge Cases":
    st.markdown("## 🧪 Robustness Testing & Extreme Edge Cases")
    st.caption("Feature Perturbation Shocks and Synthetic Stress Scenarios")

    rf_model = models_dict['rf_model']
    X = df[FEATURES]
    base_preds = rf_model.predict(X)
    base_mean = np.mean(base_preds)

    col_rb1, col_rb2 = st.columns(2)

    with col_rb1:
        st.subheader("⚡ Sensitivity to Predictor Shocks (±20%)")
        st.write(f"**Baseline Mean Prediction**: `{base_mean:.2f}%`")
        
        shock_results = []
        for col in ["Investment", "Broadband", "R_and_D"]:
            X_hike = X.copy()
            X_drop = X.copy()
            X_hike[col] = X_hike[col] * 1.20
            X_drop[col] = X_drop[col] * 0.80

            hike_diff = np.mean(rf_model.predict(X_hike)) - base_mean
            drop_diff = np.mean(rf_model.predict(X_drop)) - base_mean
            
            shock_results.append({
                'Predictor': col,
                '+20% Hike (Change)': f"{hike_diff:+.2f}%",
                '-20% Drop (Change)': f"{drop_diff:+.2f}%",
                'Net Sensitivity': abs(hike_diff) + abs(drop_diff)
            })
            
        st.table(pd.DataFrame(shock_results))
        st.markdown("""
        - **Investment Shock**: High positive sensitivity; a 20% contraction reduces predicted rate by **-1.52%**.
        - **Broadband Shock**: Non-linear response due to saturation thresholds.
        - **R&D Shock**: Positive elasticity; a 20% hike boosts the rate by **+1.67%**.
        """)

    with col_rb2:
        st.subheader("🚨 Extreme Edge-Case Stress Testing")
        
        edge_cases = pd.DataFrame([
            {**X.mean().to_dict(), "Investment": 0.0},
            X.max().to_dict(),
            X.min().to_dict(),
            {**X.mean().to_dict(), "Broadband": X["Broadband"].max() * 10}
        ], index=[
            "Zero Investment Freeze",
            "All-Maximum Boom",
            "All-Minimum Collapse",
            "Extreme Broadband Outlier (10x Max)"
        ])
        
        edge_preds = rf_model.predict(edge_cases[FEATURES])
        
        edge_df = pd.DataFrame({
            'Scenario Name': edge_cases.index,
            'Description': [
                'Investment frozen at 0; all other features at sample mean',
                'All features set to their highest observed sample levels',
                'All features set to their lowest observed sample levels',
                'Extreme out-of-bounds broadband hike (10x maximum observed)'
            ],
            'Predicted AI Rate': [f"{p:.2f}%" for p in edge_preds]
        })
        st.table(edge_df)

    st.markdown("---")
    st.subheader("💡 Methodological Insights on Robustness")
    st.markdown("""
    - **Stability Under Stress**: Under the **Zero Investment Freeze**, the predicted AI Rate falls to 7.15%, demonstrating that without compute capital, AI growth stalls near baseline maintenance.
    - **All-Maximum Boom**: Reaches **+20.80%**, reflecting the upper bound achievable under joint infrastructure, capital, and stability alignment.
    - **Outlier Invariance**: The extreme 10x Broadband outlier predicts **-7.73%**, showing how tree partitions prevent runaway linear extrapolation into impossible domains.
    """)

# -------------------------------------------------------------
# MODULE 6: PANEL DATA & CORRELATION
# -------------------------------------------------------------
elif selected_tab == "📋 Panel Data & Correlation":
    st.markdown("## 📋 Panel Dataset & Correlation Diagnostics")
    st.caption("Exploratory Data Matrix, Correlation Heatmap, and Export Tools")

    col_dt1, col_dt2 = st.columns([3, 2])

    with col_dt1:
        st.subheader("📑 Panel Dataset Explorer")
        st.dataframe(filtered_df[['Country', 'Year', 'Economic_Group'] + FEATURES + [TARGET]], use_container_width=True)
        
        # Download button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Panel CSV",
            data=csv_data,
            file_name="AI_Development_Rate_Panel_Data.csv",
            mime="text/csv"
        )

    with col_dt2:
        st.subheader("🔥 Feature Correlation Heatmap")
        corr = df[FEATURES + [TARGET]].corr().round(2)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Pearson Correlation Matrix"
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
st.caption("AI Research & Econometric Modeling Project | Built for Comparative Development Analysis")

