import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. LOAD DATASET & DEFINE STUDY PARAMETERS
# ==============================================================================
Compute_Investment = pd.read_excel(
    r"C:\Users\Junior\OneDrive\Documentos\3rd Year\ITDPA Datasets\Infrastructure\AI compute\AI Investment into compute.xlsx"
)

Start_Year = 2018
End_Year = 2024

Target_Countries = ["China (People's Republic of)", "Germany", "India", "South Africa", "Switzerland", "United States"]
Developed_Nations = ["Germany", "Switzerland", "United States"]
Developing_Nations = ["China (People's Republic of)", "India", "South Africa"]

# ==============================================================================
# 2. PANEL RECONSTRUCTION & MISSINGNESS HANDLING
# ==============================================================================
# Filter within time frame and target countries
filtered = Compute_Investment[
    (Compute_Investment["Country_label"].isin(Target_Countries)) &
    (Compute_Investment["Quarter"].between(Start_Year, End_Year))
].copy()

# Note: In raw data, missing country-years (e.g. South Africa 2018, 2019, 2023)
# are completely missing rows rather than NaN.
# Pivoting reconstructs the balanced panel (Countries x Years) and exposes true NaNs.
deal_matrix_raw = filtered.pivot(index="Country_label", columns="Quarter", values="Median_deal")

print("=== 1. Raw Median Deal Matrix (Before Imputation) ===")
print(deal_matrix_raw.round(2))

# Check missingness percentage per country
missing_report = deal_matrix_raw.isna().sum(axis=1) / deal_matrix_raw.shape[1] * 100
print("\n=== Missingness Percentage per Country (2018-2024) ===")
print(missing_report.to_frame("Missingness (%)").round(1))

# Country-specific Median Imputation across rows
deal_matrix_imputed = deal_matrix_raw.apply(lambda row: row.fillna(row.median()), axis=1)

print("\n=== 2. Completed Median Deal Matrix (After Country-Specific Median Imputation) ===")
print(deal_matrix_imputed.round(2))

# Flatten back to long format for EDA and plotting
df_clean = deal_matrix_imputed.reset_index().melt(
    id_vars="Country_label", 
    var_name="Year", 
    value_name="Median_deal_imputed"
)
df_clean["Year"] = df_clean["Year"].astype(int)
df_clean["Economic_Group"] = np.where(
    df_clean["Country_label"].isin(Developed_Nations), 
    "Developed", 
    "Developing"
)

# ==============================================================================
# 3. STATISTICAL EDA
# ==============================================================================
# Country-level summary statistics
country_stats = df_clean.groupby("Country_label")["Median_deal_imputed"].agg(
    Mean="mean",
    Median="median",
    Std="std",
    Min="min",
    Max="max",
    IQR=lambda x: np.percentile(x, 75) - np.percentile(x, 25)
).round(2)

# Absolute and % growth (2018 to 2024)
country_stats["Change (2018-2024)"] = (deal_matrix_imputed[2024] - deal_matrix_imputed[2018]).round(2)
country_stats["Growth (%)"] = ((deal_matrix_imputed[2024] - deal_matrix_imputed[2018]) / deal_matrix_imputed[2018] * 100).round(1)

print("\n=== 3. Descriptive Statistics & Growth (2018 - 2024) ===")
print(country_stats)

# Economic Group Aggregation
group_annual_avg = df_clean.groupby(["Economic_Group", "Year"])["Median_deal_imputed"].mean().unstack().round(2)
print("\n=== 4. Mean Median Deal Size by Economic Group ===")
print(group_annual_avg)

# ==============================================================================
# 4. DATA VISUALIZATIONS
# ==============================================================================
sns.set_theme(style="whitegrid", font_scale=1.05)

# Visual 1: Multi-Line Trend Chart (Best for time trajectories & growth comparison)
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_clean,
    x="Year",
    y="Median_deal_imputed",
    hue="Country_label",
    style="Economic_Group",
    markers=True,
    dashes=False,
    linewidth=2.5,
    markersize=9
)
plt.title("AI Compute Investment: Median Deal Size by Country (2018 - 2024)", fontsize=14, weight="bold", pad=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Median Deal Size (USD Millions)", fontsize=12)
plt.xticks(range(Start_Year, End_Year + 1))
plt.legend(title="Country & Category", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Visual 2: Heatmap Matrix (Best for compact intensity comparison & visual EDA)
plt.figure(figsize=(10, 5))
sns.heatmap(
    deal_matrix_imputed, 
    annot=True, 
    fmt=".1f", 
    cmap="YlGnBu", 
    cbar_kws={'label': 'Median Deal Size (USD M)'},
    linewidths=0.5
)
plt.title("Heatmap: AI Compute Median Deal Size (2018 - 2024)", fontsize=13, weight="bold", pad=12)
plt.xlabel("Year", fontsize=11)
plt.ylabel("Country", fontsize=11)
plt.tight_layout()
plt.show()

# Visual 3: Developed vs Developing Aggregate Trajectory
plt.figure(figsize=(9, 5))
group_annual_avg.T.plot(marker="o", linewidth=2.5, figsize=(9, 5))
plt.title("Economic Group Comparison: Average Median Deal Size", fontsize=13, weight="bold", pad=12)
plt.xlabel("Year", fontsize=11)
plt.ylabel("Average Median Deal Size (USD M)", fontsize=11)
plt.xticks(range(Start_Year, End_Year + 1))
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
