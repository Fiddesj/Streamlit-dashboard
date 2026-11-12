"""
Exploratory Data Analysis (EDA) and Missing Value Handling
Datasets:
1. Total population.csv
2. Individuals using the Internet.csv
3. International bandwidth usage.csv
4. Individuals with ICT skills.csv
5. API_EG.ELC.ACCS.ZS_DS2_en_excel_v2_467838.xls (World Bank Electricity Access)

Missing Value Strategy:
Follows the project's Missing Value Handling protocol:
- Assessment & Measurement (MCAR/MAR/MNAR diagnosis, % missingness calculation)
- <5% missingness: Removal or preservation of observation if representativeness holds.
- 5% - 20% missingness: KNN Imputation (preserves multivariate relationships) or Mean/Median baseline.
- 20% - 40% missingness: Multiple imputation / KNN with sensitivity assessment.
- >40% missingness: Exclusion from primary quantitative models with documentation of bias.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

VIS_DIR = "visualizations"
CLEAN_DIR = "cleaned_data"
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

FOCUS_COUNTRIES = ['China', 'Germany', 'Switzerland', 'South Africa', 'India', 'United States']
DEVELOPED = ['Germany', 'Switzerland', 'United States']
DEVELOPING = ['China', 'India', 'South Africa']

print("="*70)
print("COMPREHENSIVE EDA AND MISSING VALUE TREATMENT PIPELINE")
print("="*70)

# ==============================================================================
# 1. TOTAL POPULATION EDA
# ==============================================================================
print("\n" + "="*50)
print("1. DATASET: TOTAL POPULATION (2018 - 2024)")
print("="*50)

pop_df = pd.read_csv("Total population.csv")
print(f"Dimensions: {pop_df.shape[0]} rows, {pop_df.shape[1]} columns")
print(f"Missing Values: {pop_df.isna().sum().sum()} total across all columns")

# Summary stats
pop_stats_all = pop_df.describe()
print("\n--- Summary Statistics (All Countries, in millions) ---")
print((pop_stats_all / 1e6).round(2))

# Split into Focus Groups
pop_focus = pop_df[pop_df['Economy'].isin(FOCUS_COUNTRIES)].copy()
pop_developed = pop_df[pop_df['Economy'].isin(DEVELOPED)].copy()
pop_developing = pop_df[pop_df['Economy'].isin(DEVELOPING)].copy()

print("\n--- Focus Developed Nations Population (Millions) ---")
print(pop_developed[['Economy', '2018', '2021', '2024']].to_string(index=False))

print("\n--- Focus Developing Nations Population (Millions) ---")
print(pop_developing[['Economy', '2018', '2021', '2024']].to_string(index=False))

pop_df.to_csv(os.path.join(CLEAN_DIR, "clean_total_population.csv"), index=False)

# Visualization 1: Population Trend for Focus Nations
fig, ax = plt.subplots(figsize=(10, 6))
years = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']

colors = {
    'China': '#e41a1c', 'India': '#ff7f00', 'South Africa': '#4daf4a',
    'United States': '#377eb8', 'Germany': '#984ea3', 'Switzerland': '#a65628'
}

for country in FOCUS_COUNTRIES:
    c_data = pop_df[pop_df['Economy'] == country]
    if not c_data.empty:
        vals = c_data[years].values[0] / 1e6
        ax.plot(years, vals, marker='o', linewidth=2.2, label=country, color=colors.get(country, '#333333'))

ax.set_title("Population Trends (2018-2024) - Focus Nations (Log Scale)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Population (Millions, Log Scale)", fontsize=12)
ax.set_yscale('log')
ax.legend(title="Country", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "pop_trends_focus_countries.png"), dpi=300)
plt.close()

# Visualization 2: Distribution of Population 2024 (Boxplot + Histogram)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(pop_df['2024'] / 1e6, bins=30, kde=True, color='#2b5c8f', ax=ax1)
ax1.set_title("Global Population Distribution (2024)", fontsize=13, fontweight='bold')
ax1.set_xlabel("Population (Millions)", fontsize=11)
ax1.set_ylabel("Frequency", fontsize=11)
ax1.set_yscale('log')

sns.boxplot(y=pop_df['2024'] / 1e6, color='#6baed6', ax=ax2)
ax2.set_title("Global Population Outlier Boxplot (2024)", fontsize=13, fontweight='bold')
ax2.set_ylabel("Population (Millions)", fontsize=11)
ax2.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "pop_distribution_2024.png"), dpi=300)
plt.close()


# ==============================================================================
# 2. INDIVIDUALS USING THE INTERNET EDA & MISSING VALUE TREATMENT
# ==============================================================================
print("\n" + "="*50)
print("2. DATASET: INDIVIDUALS USING THE INTERNET (% POPULATION)")
print("="*50)

net_df = pd.read_csv("Individuals using the Internet.csv")
net_years = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']

# Clean strings to numeric float
net_numeric = net_df.copy()
for col in net_years:
    net_numeric[col] = pd.to_numeric(net_numeric[col].astype(str).str.rstrip('%').replace('-', np.nan), errors='coerce')

print(f"Dimensions: {net_numeric.shape[0]} economies, {net_numeric.shape[1]} columns")

# Missing Value Audit
print("\n--- Missingness Audit (Internet Usage) ---")
missing_net_pct = (net_numeric[net_years].isna().mean() * 100).round(2)
missing_net_cnt = net_numeric[net_years].isna().sum()
missing_audit_net = pd.DataFrame({"Missing Count": missing_net_cnt, "Missing %": missing_net_pct})
print(missing_audit_net)

# According to protocol:
# Missingness per variable is between 2.06% and 6.70% (moderately low, within the <5% and 5%-20% threshold).
# KNN Imputation (k=5) preserves multivariate temporal correlations across neighboring years.
imputer_net = KNNImputer(n_neighbors=5, weights='distance')
net_imputed_matrix = imputer_net.fit_transform(net_numeric[net_years])
net_clean_df = pd.DataFrame(net_imputed_matrix, columns=net_years, index=net_numeric.index)
net_clean_df.insert(0, 'Economy', net_numeric['Economy'])

print("\n--- Summary Statistics (Post-KNN Imputation) ---")
print(net_clean_df[net_years].describe().round(2))

# Focus Group stats
net_focus = net_clean_df[net_clean_df['Economy'].isin(FOCUS_COUNTRIES)]
print("\n--- Internet Adoption in Focus Nations (% of Population) ---")
print(net_focus[['Economy', '2018', '2021', '2024']].round(1).to_string(index=False))

net_clean_df.to_csv(os.path.join(CLEAN_DIR, "clean_individuals_using_internet.csv"), index=False)

# Visualization 3: Internet Adoption Trajectory in Focus Nations
fig, ax = plt.subplots(figsize=(10, 6))
for country in FOCUS_COUNTRIES:
    c_data = net_clean_df[net_clean_df['Economy'] == country]
    if not c_data.empty:
        vals = c_data[net_years].values[0]
        ax.plot(net_years, vals, marker='s', linewidth=2.4, label=country, color=colors.get(country, '#333333'))

ax.set_title("Internet Penetration Rate (% Population, 2018-2024)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Individuals Using Internet (%)", fontsize=12)
ax.set_ylim(0, 105)
ax.axhline(100, color='grey', linestyle='--', alpha=0.5)
ax.legend(title="Country", loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "internet_penetration_focus_countries.png"), dpi=300)
plt.close()

# Visualization 4: Global Distribution & Boxplots Over Time
fig, ax = plt.subplots(figsize=(11, 6))
net_melted = net_clean_df.melt(id_vars=['Economy'], value_vars=net_years, var_name='Year', value_name='Penetration')
sns.boxplot(x='Year', y='Penetration', data=net_melted, palette='Blues', ax=ax)
sns.stripplot(x='Year', y='Penetration', data=net_melted, color='black', alpha=0.2, jitter=0.2, size=3, ax=ax)
ax.set_title("Global Internet Penetration Distribution Shift (2018-2024)", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel("Internet Users (% of Population)", fontsize=12)
ax.set_xlabel("Year", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "internet_penetration_global_boxplot.png"), dpi=300)
plt.close()


# ==============================================================================
# 3. INTERNATIONAL BANDWIDTH USAGE EDA & MISSING VALUE TREATMENT
# ==============================================================================
print("\n" + "="*50)
print("3. DATASET: INTERNATIONAL BANDWIDTH USAGE (Mbit/s)")
print("="*50)

bw_df = pd.read_csv("International bandwidth usage.csv")
all_bw_cols = ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026']

bw_numeric = bw_df.copy()
for col in all_bw_cols:
    bw_numeric[col] = pd.to_numeric(bw_numeric[col].astype(str).replace('-', np.nan), errors='coerce')

print(f"Dimensions: {bw_numeric.shape[0]} economies, {bw_numeric.shape[1]} columns")

# Audit missingness
missing_bw_pct = (bw_numeric[all_bw_cols].isna().mean() * 100).round(2)
missing_bw_cnt = bw_numeric[all_bw_cols].isna().sum()
missing_audit_bw = pd.DataFrame({"Missing Count": missing_bw_cnt, "Missing %": missing_bw_pct})
print("\n--- Missingness Audit (Bandwidth Usage) ---")
print(missing_audit_bw)

# PROTOCOL DECISION RULE:
# 2024 (99.28%), 2025 (100%), 2026 (100%) exceed the >40% threshold.
# Per guideline: ">40%: If a key independent variable is missing for over 40% of observations,
# it will be excluded from the primary analysis. Imputing such a large proportion of data would introduce significant bias."
# Therefore, 2024-2026 are excluded from the primary statistical models.
valid_bw_years = ['2018', '2019', '2020', '2021', '2022', '2023']
print("\nExcluding uncollected years 2024, 2025, 2026 (>40% missingness). Retaining 2018-2023.")

# 2018-2023 missingness is 16.67% - 23.19% (moderate missingness, 5%-20% & 20%-40% boundary).
# Apply KNN Imputer (k=5) on log-transformed bandwidth to preserve geometric scaling across orders of magnitude.
bw_log = np.log10(bw_numeric[valid_bw_years].replace(0, np.nan))
imputer_bw = KNNImputer(n_neighbors=5, weights='distance')
bw_imputed_log = imputer_bw.fit_transform(bw_log)
bw_imputed_matrix = np.power(10, bw_imputed_log)

bw_clean_df = pd.DataFrame(bw_imputed_matrix, columns=valid_bw_years, index=bw_numeric.index)
bw_clean_df.insert(0, 'Economy', bw_numeric['Economy'])

print("\n--- Summary Statistics (Bandwidth in Gbit/s, 2018-2023) ---")
print((bw_clean_df[valid_bw_years].describe() / 1000).round(2))

# Focus countries bandwidth
bw_focus = bw_clean_df[bw_clean_df['Economy'].isin(FOCUS_COUNTRIES)]
print("\n--- Available Focus Countries Bandwidth (Gbit/s) ---")
print((bw_focus.set_index('Economy')[valid_bw_years] / 1000).round(1))

bw_clean_df.to_csv(os.path.join(CLEAN_DIR, "clean_international_bandwidth.csv"), index=False)

# Visualization 5: Bandwidth Growth Trajectory (Log Scale)
fig, ax = plt.subplots(figsize=(10, 6))
top_bw_countries = bw_clean_df.sort_values(by='2023', ascending=False).head(10)['Economy'].tolist()

for country in top_bw_countries:
    c_data = bw_clean_df[bw_clean_df['Economy'] == country]
    vals = c_data[valid_bw_years].values[0] / 1000  # Convert to Gbit/s
    ax.plot(valid_bw_years, vals, marker='o', linewidth=2, label=country)

ax.set_title("International Bandwidth Growth (Top 10 Economies, Gbit/s)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Bandwidth Capacity (Gbit/s, Log Scale)", fontsize=12)
ax.set_yscale('log')
ax.legend(title="Economy", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "bandwidth_top10_growth.png"), dpi=300)
plt.close()

# Visualization 6: Focus Countries vs Global Median Bandwidth
fig, ax = plt.subplots(figsize=(10, 6))
median_bw = bw_clean_df[valid_bw_years].median() / 1000
ax.plot(valid_bw_years, median_bw, linestyle='--', color='black', linewidth=2.5, label='Global Median')

for country in ['China', 'India', 'South Africa']:
    c_data = bw_clean_df[bw_clean_df['Economy'] == country]
    if not c_data.empty:
        vals = c_data[valid_bw_years].values[0] / 1000
        ax.plot(valid_bw_years, vals, marker='o', linewidth=2.5, label=country, color=colors.get(country))

ax.set_title("Bandwidth Capacity: Focus Developing Nations vs Global Median (Gbit/s)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Bandwidth (Gbit/s, Log Scale)", fontsize=12)
ax.set_yscale('log')
ax.legend(title="Legend", loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "bandwidth_focus_vs_median.png"), dpi=300)
plt.close()


# ==============================================================================
# 4. ELECTRICITY ACCESS (WORLD BANK) EDA & MISSING VALUE TREATMENT
# ==============================================================================
print("\n" + "="*50)
print("4. DATASET: ACCESS TO ELECTRICITY (% OF POPULATION - WORLD BANK)")
print("="*50)

excel_file = "API_EG.ELC.ACCS.ZS_DS2_en_excel_v2_467838.xls"
elec_df = pd.read_excel(excel_file, sheet_name="Data", header=3)
meta_country = pd.read_excel(excel_file, sheet_name="Metadata - Countries")

elec_years = [str(y) for y in range(2000, 2025)]
analysis_elec_years = [str(y) for y in range(2018, 2025)]

print(f"Dimensions: {elec_df.shape[0]} entities, {elec_df.shape[1]} columns")

# Missing Value Audit
print("\n--- Missingness Audit (Electricity Access 2018-2025) ---")
missing_elec_pct = (elec_df[[str(y) for y in range(2018, 2026)]].isna().mean() * 100).round(2)
missing_elec_cnt = elec_df[[str(y) for y in range(2018, 2026)]].isna().sum()
missing_audit_elec = pd.DataFrame({"Missing Count": missing_elec_cnt, "Missing %": missing_elec_pct})
print(missing_audit_elec)

# PROTOCOL DECISION RULE:
# 2025 is 100% missing (>40%), so 2025 is excluded.
# For 2018-2024, missingness is only 1.13% (3 out of 265 entities: American Samoa, Kosovo, Not classified).
# Per protocol: "<5% missingness: removal of the observation will be considered if cases do not form a systematic pattern."
# In this dataset, the 3 entities missing all years are removed from primary analysis to preserve sample representativeness.
elec_clean_df = elec_df.dropna(subset=analysis_elec_years).copy()

# Merge IncomeGroup and Region metadata
elec_clean_df = elec_clean_df.merge(
    meta_country[['Country Code', 'Region', 'IncomeGroup']],
    on='Country Code',
    how='left'
)

print(f"Observations retained post-<5% exclusion: {len(elec_clean_df)} / {len(elec_df)}")
print("\n--- Summary Statistics: Access to Electricity (2018-2024) ---")
print(elec_clean_df[analysis_elec_years].describe().round(2))

# Income group breakdown for 2024
print("\n--- Electricity Access by World Bank Income Group (2024 Mean & Median) ---")
income_grp_stats = elec_clean_df.groupby('IncomeGroup')['2024'].agg(['count', 'mean', 'median', 'std', 'min']).round(2)
print(income_grp_stats)

# Focus countries
elec_focus = elec_clean_df[elec_clean_df['Country Name'].isin(FOCUS_COUNTRIES)]
print("\n--- Access to Electricity in Focus Countries (% of Population) ---")
print(elec_focus[['Country Name', 'IncomeGroup', '2018', '2021', '2024']].round(1).to_string(index=False))

elec_clean_df.to_csv(os.path.join(CLEAN_DIR, "clean_electricity_access.csv"), index=False)

# Visualization 7: Access to Electricity by Income Group (2024 Boxplot)
fig, ax = plt.subplots(figsize=(10, 6))
order = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
sns.boxplot(
    x='IncomeGroup', y='2024', data=elec_clean_df.dropna(subset=['IncomeGroup']),
    order=order, palette='viridis', ax=ax
)
sns.stripplot(
    x='IncomeGroup', y='2024', data=elec_clean_df.dropna(subset=['IncomeGroup']),
    order=order, color='black', alpha=0.3, jitter=0.2, ax=ax
)
ax.set_title("Electrification Access Rate by Income Group (2024)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Income Group", fontsize=12)
ax.set_ylabel("Access to Electricity (% Population)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "electricity_income_group_2024.png"), dpi=300)
plt.close()

# Visualization 8: Electrification Trends for Focus Nations (2000-2024)
fig, ax = plt.subplots(figsize=(11, 6))
trend_years = [str(y) for y in range(2000, 2025, 2)]
for country in FOCUS_COUNTRIES:
    c_data = elec_clean_df[elec_clean_df['Country Name'] == country]
    if not c_data.empty:
        vals = c_data[trend_years].values[0]
        ax.plot(trend_years, vals, marker='o', linewidth=2.4, label=country, color=colors.get(country, '#333333'))

ax.set_title("Electricity Access Trajectory (2000-2024) - Focus Nations", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Access to Electricity (% of Population)", fontsize=12)
ax.set_ylim(50, 103)
ax.legend(title="Country", loc="lower right", frameon=True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "electricity_focus_trajectory_2000_2024.png"), dpi=300)
plt.close()


# ==============================================================================
# 5. INDIVIDUALS WITH ICT SKILLS EDA & MISSING VALUE TREATMENT
# ==============================================================================
print("\n" + "="*50)
print("5. DATASET: INDIVIDUALS WITH ICT SKILLS")
print("="*50)

ict_df = pd.read_csv("Individuals with ICT skills.csv")
ict_years = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']

ict_numeric = ict_df.copy()
for col in ict_years:
    ict_numeric[col] = pd.to_numeric(ict_numeric[col].astype(str).str.rstrip('%').replace('-', np.nan), errors='coerce')

print(f"Dimensions: {ict_numeric.shape[0]} rows across {ict_numeric['Economy'].nunique()} economies")

# Missing Value Audit
print("\n--- Missingness Audit (ICT Skills by Year) ---")
missing_ict_pct = (ict_numeric[ict_years].isna().mean() * 100).round(2)
missing_ict_cnt = ict_numeric[ict_years].isna().sum()
missing_audit_ict = pd.DataFrame({"Missing Count": missing_ict_cnt, "Missing %": missing_ict_pct})
print(missing_audit_ict)

# PROTOCOL DECISION RULE:
# 2018-2020 are 100% missing (>40%), so they are completely excluded.
# 2021 (67.71%), 2022 (76.00%), 2024 (61.74%) also exceed 40% missingness globally across all categories.
# Per protocol: Variables with >40% missingness must be excluded from primary comparative models to avoid introducing extreme bias.
# However, 2023 offers the richest cross-sectional coverage across skill categories.
# Let's inspect category-specific coverage in 2023:
print("\n--- 2023 Coverage by Skill Category ---")
for cat in ict_numeric['Category'].unique():
    sub = ict_numeric[ict_numeric['Category'] == cat]
    pct_avail = (sub['2023'].notna().mean() * 100).round(1)
    print(f"  {cat}: {sub['2023'].notna().sum()}/{len(sub)} reporting ({pct_avail}% coverage)")

# Extract 2023 cross-sectional skill matrix
ict_2023 = ict_numeric[['Economy', 'Category', 'Segment', '2023']].dropna(subset=['2023']).copy()
ict_2023.to_csv(os.path.join(CLEAN_DIR, "clean_ict_skills_2023_available.csv"), index=False)

# Summary statistics for available 2023 skill data by Segment
print("\n--- 2023 ICT Skill Share Distribution by Segment (% of Population) ---")
print(ict_2023.groupby('Segment')['2023'].describe().round(2))

# Visualization 9: Missingness Heatmap / Bar Chart across Datasets
fig, ax = plt.subplots(figsize=(10, 5))
audit_summary = pd.DataFrame({
    'Electricity (2018-24)': 1.13,
    'Population (2018-24)': 0.0,
    'Internet (2018-24)': missing_net_pct.mean(),
    'Bandwidth (2018-23)': missing_bw_pct[valid_bw_years].mean(),
    'Bandwidth (2024-26)': 99.76,
    'ICT Skills (2018-20)': 100.0,
    'ICT Skills (2023)': missing_ict_pct['2023']
}, index=['Missingness %']).T

colors_audit = ['#2ca02c' if v < 5 else ('#ff7f0e' if v <= 40 else '#d62728') for v in audit_summary['Missingness %']]
bars = ax.barh(audit_summary.index, audit_summary['Missingness %'], color=colors_audit, edgecolor='black', alpha=0.85)
ax.axvline(5, color='green', linestyle='--', linewidth=1.5, label='<5% Removal / Complete')
ax.axvline(20, color='orange', linestyle='--', linewidth=1.5, label='5-20% Mean/KNN Imputation')
ax.axvline(40, color='red', linestyle='--', linewidth=1.5, label='>40% Exclusion Threshold')
ax.set_title("Missing Value Diagnostic & Decision Protocol Mapping", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Percentage Missing (%)", fontsize=12)
ax.set_xlim(0, 105)
ax.legend(loc='lower right', frameon=True)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "missingness_protocol_audit.png"), dpi=300)
plt.close()

# Visualization 10: 2023 ICT Skill Distribution by Category and Segment
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='Category', y='2023', hue='Segment', data=ict_2023[ict_2023['Category'] != 'Overall ICT skill level'], palette='Set2', ax=ax)
ax.set_title("ICT Skill Level Distribution by Competency Domain (2023 Valid Sample)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Competency Category", fontsize=12)
ax.set_ylabel("Share of Population (%)", fontsize=12)
plt.xticks(rotation=20, ha='right')
ax.legend(title="Proficiency Segment", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "ict_skills_distribution_2023.png"), dpi=300)
plt.close()


# ==============================================================================
# 6. CROSS-DATASET SYNTHESIS: INFRASTRUCTURE & READINESS IN FOCUS NATIONS
# ==============================================================================
print("\n" + "="*50)
print("6. CROSS-DATASET SYNTHESIS: DIGITAL READINESS INDEX (2023/2024)")
print("="*50)

# Build unified comparative matrix for 2023 across available metrics
synthesis_records = []
for c in FOCUS_COUNTRIES:
    # Pop 2023
    p = pop_df[pop_df['Economy'] == c]['2023'].values[0] if not pop_df[pop_df['Economy'] == c].empty else np.nan
    # Internet 2023 (imputed)
    net_val = net_clean_df[net_clean_df['Economy'] == c]['2023'].values[0] if not net_clean_df[net_clean_df['Economy'] == c].empty else np.nan
    # Bandwidth 2023 (imputed)
    bw_val = bw_clean_df[bw_clean_df['Economy'] == c]['2023'].values[0] if not bw_clean_df[bw_clean_df['Economy'] == c].empty else np.nan
    # Elec 2023
    elec_val = elec_clean_df[elec_clean_df['Country Name'] == c]['2023'].values[0] if not elec_clean_df[elec_clean_df['Country Name'] == c].empty else np.nan
    
    # Calculate per capita bandwidth (kbit/s per person)
    bw_per_capita = (bw_val * 1e6) / p if (pd.notna(bw_val) and pd.notna(p)) else np.nan
    
    synthesis_records.append({
        'Country': c,
        'Group': 'Developed' if c in DEVELOPED else 'Developing',
        'Population_2023_M': round(p / 1e6, 2) if pd.notna(p) else np.nan,
        'Internet_Penetration_%': round(net_val, 2) if pd.notna(net_val) else np.nan,
        'Electricity_Access_%': round(elec_val, 2) if pd.notna(elec_val) else np.nan,
        'Total_Bandwidth_Gbps': round(bw_val / 1000, 2) if pd.notna(bw_val) else np.nan,
        'Bandwidth_Per_Capita_kbps': round(bw_per_capita, 2) if pd.notna(bw_per_capita) else np.nan
    })

synth_df = pd.DataFrame(synthesis_records)
print("\n--- Digital Infrastructure Synthesis Matrix (Focus Nations, 2023) ---")
print(synth_df.to_string(index=False))
synth_df.to_csv(os.path.join(CLEAN_DIR, "cross_dataset_synthesis_focus_countries.csv"), index=False)

# Visualization 11: Multi-Metric Comparison (Electricity vs Internet vs Population)
fig, ax = plt.subplots(figsize=(10, 6))
# Join global dataset of 2023 Internet and Electricity
combined_global = net_clean_df[['Economy', '2023']].rename(columns={'2023': 'Internet_2023'}).merge(
    elec_clean_df[['Country Name', '2023', 'IncomeGroup']].rename(columns={'Country Name': 'Economy', '2023': 'Electricity_2023'}),
    on='Economy',
    how='inner'
).merge(
    pop_df[['Economy', '2023']].rename(columns={'2023': 'Pop_2023'}),
    on='Economy',
    how='inner'
)

sns.scatterplot(
    data=combined_global,
    x='Electricity_2023',
    y='Internet_2023',
    hue='IncomeGroup',
    size='Pop_2023',
    sizes=(40, 600),
    alpha=0.7,
    palette='Set1',
    ax=ax
)

# Label focus nations
for c in FOCUS_COUNTRIES:
    row = combined_global[combined_global['Economy'] == c]
    if not row.empty:
        ax.annotate(
            c,
            (row['Electricity_2023'].values[0], row['Internet_2023'].values[0]),
            xytext=(7, 5), textcoords='offset points',
            fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5)
        )

ax.set_title("Global Digital Readiness: Electricity Access vs Internet Penetration (2023)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Access to Electricity (% Population)", fontsize=12)
ax.set_ylabel("Internet Users (% Population)", fontsize=12)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "digital_readiness_scatter_global.png"), dpi=300)
plt.close()

# Visualization 12: Group Comparison Bar Chart (Developed vs Developing)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Internet & Electricity comparison
synth_melted = synth_df.melt(
    id_vars=['Country', 'Group'],
    value_vars=['Internet_Penetration_%', 'Electricity_Access_%'],
    var_name='Metric',
    value_name='Value'
)
sns.barplot(data=synth_melted, x='Country', y='Value', hue='Metric', palette='coolwarm', ax=ax1)
ax1.set_title("Internet Penetration & Electricity Access (2023)", fontsize=13, fontweight='bold')
ax1.set_ylabel("Percentage (%)", fontsize=11)
ax1.set_xlabel("")
ax1.set_ylim(0, 115)
ax1.tick_params(axis='x', rotation=30)
ax1.legend(loc='lower right', frameon=True)

# Bandwidth per capita comparison (where available)
synth_bw = synth_df.dropna(subset=['Bandwidth_Per_Capita_kbps'])
sns.barplot(data=synth_bw, x='Country', y='Bandwidth_Per_Capita_kbps', hue='Group', palette='muted', ax=ax2)
ax2.set_title("Bandwidth per Capita (kbps/person, 2023)", fontsize=13, fontweight='bold')
ax2.set_ylabel("Bandwidth per Capita (kbit/s)", fontsize=11)
ax2.set_xlabel("")
ax2.tick_params(axis='x', rotation=30)
ax2.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "focus_countries_infrastructure_comparison.png"), dpi=300)
plt.close()

print("\n" + "="*70)
print(f"EDA AND VISUALIZATIONS GENERATED SUCCESSFULLY IN '{VIS_DIR}/'!")
print(f"CLEANED DATASETS SAVED TO '{CLEAN_DIR}/'!")
print("="*70)
