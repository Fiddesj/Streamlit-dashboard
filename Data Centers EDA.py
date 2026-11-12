import numpy as np
import pandas as pd

#load data frame
Data_centers_total= pd.read_excel(r"C:\Users\Junior\OneDrive\Documentos\3rd Year\ITDPA Datasets\Infrastructure\Data centers\Global_Data_Centers_2026.xlsx")

image= Data_centers_total.head()

print(image)

# isolate developing nations as a group
Developing_nations = Data_centers_total.iloc[[2, 3, 5]].copy()

#isolate developed nations as a group
Developed_nations = Data_centers_total.iloc[[0, 1, 4]].copy()

#Descriptive statistics
#All the countries
Collective_Statistics= Data_centers_total.describe()
print("\n The Basic statstics for all the countries ")
print(Collective_Statistics)
#Developing nations 
Developing_nations_statistics = Developing_nations.describe()
print("\n The basic statistics for the developing nations ")
print(Developing_nations_statistics)
#Developed nations
Developed_nations_statistics = Developed_nations.describe()
print("\n The basic statistics for the developed nations ")
print(Developed_nations_statistics)


