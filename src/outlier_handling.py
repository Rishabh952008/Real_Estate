import pandas as pd 
import numpy as np

df = pd.read_csv('prepared_data/gurgaon_properties_cleaned_v2.csv')

df.drop_duplicates()

# Calculate the IQR for the 'price' column
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]

# Calculate the IQR for the 'price' column
Q1 = df['price_per_sqft'].quantile(0.25)
Q3 = df['price_per_sqft'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers_sqft = df[(df['price_per_sqft'] < lower_bound) | (df['price_per_sqft'] > upper_bound)]

outliers_sqft['area'] = outliers_sqft['area'].apply(lambda x:x*9 if x<1000 else x)

outliers_sqft['price_per_sqft'] = round((outliers_sqft['price']*10000000)/outliers_sqft['area'])

df.update(outliers_sqft)

df = df[df['price_per_sqft'] <= 50000]

df = df[df['area'] < 50000]


df = df[df['price_per_sqft']>100]
# 818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471

df.to_csv('prepared_data/outliers_cleaned.csv',index=False)