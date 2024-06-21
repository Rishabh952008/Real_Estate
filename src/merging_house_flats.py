import pandas as pd
import numpy as np

flats = pd.read_csv('prepared_data/flats_cleaned.csv')
houses = pd.read_csv('prepared_data/houses_cleaned.csv')

df = pd.concat([flats,houses],ignore_index=True)

df = df.sample(df.shape[0],ignore_index=True)


# Import Path class
from pathlib import Path
 
# Path
path = 'prepared_data/gurgaon_properties_merged.csv'
 
# Instantiate the Path class
obj = Path(path)
 
# Check if path points to
# an existing file or directory
# print(obj.exists())
df.to_csv(path,index=False)
