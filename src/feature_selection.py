import numpy as np
import pandas as pd





df = pd.read_csv('prepared_data/missing_value_imputed.csv')

# actually i thought that price_per_sqft and society should not be asked by the user 

train_df = df.drop(columns=['society','price_per_sqft'])

def categorize_luxury(score):
    if 0 <= score < 50:
        return "Low"
    elif 50 <= score < 150:
        return "Medium"
    elif 150 <= score <= 175:
        return "High"
    else:
        return None  # or "Undefined" or any other label for scores outside the defined bins
    
train_df['luxury_category'] = train_df['luxury_score'].apply(categorize_luxury)

def categorize_floor(floor):
    if 0 <= floor <= 2:
        return "Low Floor"
    elif 3 <= floor <= 10:
        return "Mid Floor"
    elif 11 <= floor <= 60:
        return "High Floor"
    else:
        return None  # or "Undefined" or any other label for floors outside the defined bins
    
train_df['floor_category'] = train_df['floorNum'].apply(categorize_floor)

train_df.drop(columns=['floorNum','luxury_score'],inplace=True)

export_df = train_df.drop(columns=['pooja room', 'study room', 'others'])

export_df.to_csv('prepared_data/post_feature_selection.csv', index=False)