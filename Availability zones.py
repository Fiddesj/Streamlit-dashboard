import numpy as np
import pandas as pd 

#Load the datasets
China_AZ_clouds= pd.read_excel(r"C:\Users\Junior\OneDrive\Documentos\3rd Year\ITDPA Datasets\Infrastructure\AI compute\China AI Compute.xlsx")
Germany_AZ_clouds= pd.read_excel(r"C:\Users\Junior\OneDrive\Documentos\3rd Year\ITDPA Datasets\Infrastructure\AI compute\Germany AI compute.xlsx")
Switzerland_AZ_clouds= pd.read_excel(r"C:\Users\Junior\OneDrive\Documentos\3rd Year\ITDPA Datasets\Infrastructure\AI compute\Switzerland AI Compute.xlsx")
South_Africa_AZ_clouds= pd.read_excel(r"C:\Users\Junior\OneDrive\Documentos\3rd Year\ITDPA Datasets\Infrastructure\AI compute\South Africa AI Compute.xlsx")
India_AZ_clouds= pd.read_excel(r"C:\Users\Junior\OneDrive\Documentos\3rd Year\ITDPA Datasets\Infrastructure\AI compute\India AI compute.xlsx")
United_States_AZ_clouds= pd.read_excel(r"C:\Users\Junior\OneDrive\Documentos\3rd Year\ITDPA Datasets\Infrastructure\AI compute\United States AI Compute.xlsx")

#Create data frames for the five tiers
#Extract rows
Tier_1_china= China_AZ_clouds.iloc[0].copy()
Tier_no_gpu_AZ_china= China_AZ_clouds.iloc[1].copy()
Tier_2_china= China_AZ_clouds.iloc[2].copy()
Tier_3_china= China_AZ_clouds.iloc[3].copy()
Tier_4_china= China_AZ_clouds.iloc[4].copy()

Tier_1_germany= Germany_AZ_clouds.iloc[0].copy()
Tier_no_gpu_AZ_germany= Germany_AZ_clouds.iloc[1].copy()
Tier_2_germany= Germany_AZ_clouds.iloc[2].copy()
Tier_3_germany= Germany_AZ_clouds.iloc[3].copy()
Tier_4_germany= Germany_AZ_clouds.iloc[4].copy()

Tier_1_switzerland= Switzerland_AZ_clouds.iloc[0].copy()
Tier_no_gpu_AZ_switzerland= Switzerland_AZ_clouds.iloc[1].copy()
Tier_2_switzerland= Switzerland_AZ_clouds.iloc[2].copy()
Tier_3_switzerland= Switzerland_AZ_clouds.iloc[3].copy()
Tier_4_switzerland= Switzerland_AZ_clouds.iloc[4].copy()

Tier_1_south_africa= South_Africa_AZ_clouds.iloc[0].copy()
Tier_no_gpu_AZ_south_africa= South_Africa_AZ_clouds.iloc[1].copy()
Tier_2_south_africa= South_Africa_AZ_clouds.iloc[2].copy()
Tier_3_south_africa= South_Africa_AZ_clouds.iloc[3].copy()
Tier_4_south_africa= South_Africa_AZ_clouds.iloc[4].copy()

Tier_1_india= India_AZ_clouds.iloc[0].copy()
Tier_no_gpu_AZ_india= India_AZ_clouds.iloc[1].copy()
Tier_2_india= India_AZ_clouds.iloc[2].copy()
Tier_3_india= India_AZ_clouds.iloc[3].copy()
Tier_4_india= India_AZ_clouds.iloc[4].copy()

Tier_1_united_states= United_States_AZ_clouds.iloc[0].copy()
Tier_no_gpu_AZ_united_states= United_States_AZ_clouds.iloc[1].copy()
Tier_2_united_states= United_States_AZ_clouds.iloc[2].copy()
Tier_3_united_states= United_States_AZ_clouds.iloc[3].copy()
Tier_4_united_states= United_States_AZ_clouds.iloc[4].copy()

#Merge the same tiers
Tier_1_clouds = pd.concat([Tier_1_china, Tier_1_germany, Tier_1_switzerland, Tier_1_south_africa, Tier_1_india, Tier_1_united_states], axis=1)
Tier_no_gpu_AZ_clouds = pd.concat([Tier_no_gpu_AZ_china, Tier_no_gpu_AZ_germany, Tier_no_gpu_AZ_switzerland, Tier_no_gpu_AZ_south_africa, Tier_no_gpu_AZ_india, Tier_no_gpu_AZ_united_states], axis=1)
Tier_2_clouds = pd.concat([Tier_2_china, Tier_2_germany, Tier_2_switzerland, Tier_2_south_africa, Tier_2_india, Tier_2_united_states], axis=1)
Tier_3_clouds = pd.concat([Tier_3_china, Tier_3_germany, Tier_3_switzerland, Tier_3_south_africa, Tier_3_india, Tier_3_united_states], axis=1)
Tier_4_clouds = pd.concat([Tier_4_china, Tier_4_germany, Tier_4_switzerland, Tier_4_south_africa, Tier_4_india, Tier_4_united_states], axis=1)

#Print the dataframes
print(Tier_1_clouds)
print(Tier_no_gpu_AZ_clouds)
print(Tier_2_clouds)
print(Tier_3_clouds)
print(Tier_4_clouds)

#basic statistics 
Tier_1_statistics= Tier_1_clouds.describe()
print("\n The basic statistics for the tier 1 clouds ")
print(Tier_1_statistics)

Tier_no_gpu_statistics= Tier_no_gpu_AZ_clouds.describe()
print("\n The basic statistics for the tier no gpu clouds ")
print(Tier_no_gpu_statistics)

Tier_2_statistics= Tier_2_clouds.describe()
print("\n The basic statistics for the tier 2 clouds ")
print(Tier_2_statistics)

Tier_3_statistics= Tier_3_clouds.describe()
print("\n The basic statistics for the tier 3 clouds ")
print(Tier_3_statistics)

Tier_4_statistics= Tier_4_clouds.describe()
print("\n The basic statistics for the tier 4 clouds ")
print(Tier_4_statistics)

#calculate averages for each tier
Tier_1_average= Tier_1_clouds.mean(axis=0)
print("\n The average for the tier 1 clouds ")
print(Tier_1_average)

Tier_no_gpu_average= Tier_no_gpu_AZ_clouds.mean(axis=0)
print("\n The average for the tier no gpu clouds ")
print(Tier_no_gpu_average)

Tier_2_average= Tier_2_clouds.mean(axis=0)
print("\n The average for the tier 2 clouds ")
print(Tier_2_average)

Tier_3_average= Tier_3_clouds.mean(axis=0)
print("\n The average for the tier 3 clouds ")
print(Tier_3_average)

Tier_4_average= Tier_4_clouds.mean(axis=0)
print("\n The average for the tier 4 clouds ")
print(Tier_4_average)

# Export results to CSV for analysis
# 1. Combined clean dataset containing all countries and their tier values
all_countries_df = pd.concat([
    China_AZ_clouds,
    Germany_AZ_clouds,
    Switzerland_AZ_clouds,
    South_Africa_AZ_clouds,
    India_AZ_clouds,
    United_States_AZ_clouds
], ignore_index=True)

all_countries_df.to_csv("availability_zones_all_countries.csv", index=False)
print("\nSaved combined data to 'availability_zones_all_countries.csv'")

# 2. Pivot table (Tiers as rows, Countries as columns) with value counts
pivot_df = all_countries_df.pivot(index='variable', columns='Country_label', values='value').fillna(0)
pivot_df.to_csv("availability_zones_matrix.csv")
print("Saved country comparison matrix to 'availability_zones_matrix.csv'")
