import numpy as np
import pandas as pd

df = pd.read_csv('prepared_data/outliers_cleaned.csv')

def handling_various_areatype(df):
    
    all_present_df = df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))]
    
    # super_to_built_up_ratio = (all_present_df['super_built_up_area']/all_present_df['built_up_area']).median()
    
    # carpet_to_built_up_ratio = (all_present_df['carpet_area']/all_present_df['built_up_area']).median()
    
    # both present built up null
    sbc_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]

    sbc_df['built_up_area'].fillna(round(((sbc_df['super_built_up_area']/1.105) + (sbc_df['carpet_area']/0.9))/2),inplace=True)
    
    df.update(sbc_df)
    
    # sb present c is null built up null
    sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]

    sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area']/1.105),inplace=True)
    
    df.update(sb_df)
    
    # sb null c is present built up null
    c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]

    c_df['built_up_area'].fillna(round(c_df['carpet_area']/0.9),inplace=True)
    
    df.update(c_df)
    
    df = df[(df['built_up_area']<13000)]
    
    anamoly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 5)][['price','area','built_up_area']]
    
    df.update(anamoly_df)
    
    df.drop(columns=['area','areaWithType','super_built_up_area','carpet_area'],inplace=True)
    
    return df

def bal(count):
    if count.split(' ')[0]=="3+":
        return 4
    elif count.split(' ')[0]=="No":
        return 0
    else:
        return int(count.split(' ')[0])
        
# Calculate the new values for floorNum where it is NaN
def calculate_floorNum(row):
    return (row['bedRoom']*400 + row['bathroom']*150 + row['balcony']*100)/row['built_up_area']


def mode_based_imputation(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']
    

def mode_based_imputation2(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']

def mode_based_imputation3(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']

def handling_floorNum(df):
    
    df['balcony']= df['balcony'].apply(bal)
    
    # Apply the calculation only to rows where floorNum is NaN and the property type is 'House'
    df['floorNum'] = df.apply(lambda row: calculate_floorNum(row) if pd.isna(row['floorNum']) and row['property_type'] == 'house' else row['floorNum'], axis=1)

    df = df[~df['floorNum'].isna()]
    
    df.drop(columns=['facing'],inplace=True)
    
    df['agePossession'] = df.apply(mode_based_imputation,axis=1)
    
    df['agePossession'] = df.apply(mode_based_imputation2,axis=1)
    
    df['agePossession'] = df.apply(mode_based_imputation3,axis=1)
    
    return df
    
df = handling_various_areatype(df=df)

df = handling_floorNum(df)

df.to_csv('prepared_data/missing_value_imputed.csv',index=False)
    