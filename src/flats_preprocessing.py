import pandas as pd
import numpy as np 
import re
from pathlib import Path

df = pd.read_csv('data/raw/flats.csv')

# Columns to drop -> property_name, link, property_id
df.drop(columns=['link','property_id'], inplace=True)

# rename columns
df.rename(columns={'area':'price_per_sqft'},inplace=True)

# cleaning society column text using regex
df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()

df = df[df['price'] != 'Price on Request']

def treat_price(x):
    if type(x) == float:
        return x
    else:
        val = x.split(" ")
        if val[1] == 'Lac':
            return round(float(val[0])/100,2)
        else:
            return round(float(val[0]),2)
        
df = df[df['price']!='price']

df['price'].dropna(inplace=True)

df = df.dropna(subset=['price'])

df['price'] = df['price'].apply(treat_price)

df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')

df = df[~df['bedRoom'].isnull()]

df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')

df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')

df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')

df.fillna({'additionalRoom':'not available'},inplace=True)

df['additionalRoom'] = df['additionalRoom'].str.lower()

df['floorNum'] = df['floorNum'].str.split(' ').str.get(0).replace('Ground','0').str.replace('Basement','-1').str.replace('Lower','0').str.extract(r'(\d+)')

#df['facing'].fillna('NA',inplace=True)
df['facing'] = df['facing'].fillna('NA')

df.insert(loc=4,column='area',value=round((df['price']*10000000)/df['price_per_sqft']))


df.insert(loc=1,column='property_type',value='flat')


 
# Path
path = 'data/processed/flats_cleaned.csv'
 
# Instantiate the Path class
obj = Path(path)
 
# Check if path points to
# an existing file or directory
# print(obj.exists())
df.to_csv(path,index=False)