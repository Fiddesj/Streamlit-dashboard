import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD & FILTER DATA
# ==========================================
HPC_World = pd.read_csv(r"C:\Users\Junior\OneDrive\Documentos\3rd Year\ITDPA Datasets\Infrastructure\Data centers\T500_TL.csv")

Target_Countries = ["Switzerland", "Germany", "United States", "China", "India", "South Africa"]
Developing_nations = ["China", "India", "South Africa"]
Developed_Nations = ["Switzerland", "Germany", "United States"]

Start_year = 2018
End_year = 2024

# The column with the HPC counts is 'OBS_VALUE'
# The column with the years is 'TIME_PERIOD'
# The column with the country name is 'REF_AREA_LABEL'
key_cols = ["REF_AREA_LABEL", "TIME_PERIOD", "OBS_VALUE"]

# Filter for target countries and years
Target_Countries_data = HPC_World[
    (HPC_World["REF_AREA_LABEL"].isin(Target_Countries)) & 
    (HPC_World["TIME_PERIOD"].between(Start_year, End_year))
][key_cols].copy()

# Add economic group categorization for comparative analysis
Target_Countries_data["Group"] = np.where(
    Target_Countries_data["REF_AREA_LABEL"].isin(Developed_Nations), 
    "Developed", 
    "Developing"
)


# 2. EDA: RESHAPING & COMPARISON TABLES

print("=== 1. Summary of Isolated HPC Data (Top 5 rows) ===")
print(Target_Countries_data.head())

# Pivot Table: Countries as rows, Years as columns, HPC counts as values
hpc_pivot = Target_Countries_data.pivot(index="REF_AREA_LABEL", columns="TIME_PERIOD", values="OBS_VALUE").fillna(0)
print("\n=== 2. HPC Count by Country Across Years (2018 - 2024) ===")
print(hpc_pivot)

# Group Summary: Developed vs Developing totals per year
group_summary = Target_Countries_data.groupby(["Group", "TIME_PERIOD"])["OBS_VALUE"].sum().unstack()
print("\n=== 3. Total HPC Count by Economic Group ===")
print(group_summary)

# Key Metrics: 2018 vs 2024 absolute change & average
hpc_pivot["Change (2018-2024)"] = hpc_pivot[2024] - hpc_pivot[2018]
hpc_pivot["Average (2018-2024)"] = hpc_pivot.loc[:, Start_year:End_year].mean(axis=1).round(2)
print("\n=== 4. Country Performance Summary ===")
print(hpc_pivot[["Change (2018-2024)", "Average (2018-2024)"]])

# ==========================================
# 3. VISUALIZATIONS
# ==========================================
sns.set_theme(style="whitegrid")

# Figure 1: Time Series Multi-Line Plot (Best for Trends & Divergence)
plt.figure(figsize=(11, 6))
palette = sns.color_palette("tab10", len(Target_Countries))
sns.lineplot(
    data=Target_Countries_data, 
    x="TIME_PERIOD", 
    y="OBS_VALUE", 
    hue="REF_AREA_LABEL", 
    style="Group",
    markers=True, 
    dashes=False, 
    linewidth=2.5,
    markersize=8
)
plt.title("HPC Count Comparison Across Target Nations (2018 - 2024)", fontsize=14, weight="bold")
plt.xlabel("Year", fontsize=12)
plt.ylabel("HPC Count (OBS_VALUE - Top 500 Supercomputers)", fontsize=12)
plt.xticks(range(Start_year, End_year + 1))
plt.legend(title="Country & Category", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Figure 2: Developed vs Developing Aggregation (Best for Group Disparity)
plt.figure(figsize=(9, 5))
group_summary.T.plot(kind="bar", figsize=(10, 5), width=0.7)
plt.title("Developed vs Developing Nations: Total HPC Share (2018 - 2024)", fontsize=13, weight="bold")
plt.xlabel("Year", fontsize=11)
plt.ylabel("Total Supercomputers", fontsize=11)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
