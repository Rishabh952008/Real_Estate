import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

st.set_page_config(page_title="Viz Demo")


df = pd.read_csv('notebooks/Model_Selection/properties.csv')

df = df[~(df['floor_category'].isnull())]

with open('models/xgb_model.pkl','rb') as file:
    xgb_model = pickle.load(file)


st.header('Enter your inputs')

# property_type
property_type = st.selectbox('Property Type',['flat','house'])

# sector
sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))

bedrooms = float(st.selectbox('Number of Bedroom',sorted(df['bedRoom'].unique().tolist())))

bathroom = float(st.selectbox('Number of Bathrooms',sorted(df['bathroom'].unique().tolist())))

balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))

property_age = st.selectbox('Property Age',sorted(df['agePossession'].unique().tolist()))

built_up_area = float(st.number_input('Built Up Area'))

servant_room = float(st.selectbox('Servant Room',[0.0, 1.0]))
store_room = float(st.selectbox('Store Room',[0.0, 1.0]))

furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique().tolist()))
luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))
floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))


def transform_input(df):
    df = df[~(df['floor_category'].isnull())]

    
    # Create the encoder
    encoder = OneHotEncoder(sparse_output=False)

    # Assuming that 'df' is your DataFrame and 'column_to_encode' is the column you want to encode
    encoded_columns = encoder.fit_transform(df[['sector','floor_category']])

    # The result is a numpy array of encoded columns
    
    df.reset_index(drop=True, inplace=True)
    # Assuming that 'df' is your DataFrame and 'encoded_columns' is the one-hot encoded numpy array

    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out())

    # Concatenate the original DataFrame and the encoded DataFrame
    df = pd.concat([df, encoded_df], axis=1)    
    
    # Check for any null values
    if df.isnull().values.any():
        print("Null values found in the DataFrame after encoding:")
        
    df =df.drop(columns=['sector','floor_category'])
    
    def encode_furnish(ftype):
        if ftype=='furnished':
          return 2.32
        elif ftype=='semifurnished':
          return 2.10
        else:
          return 1.30
    
    df['furnishing_type']=df['furnishing_type'].apply(encode_furnish)
    
    def encode_luxury(ltype):
       if ltype=='High':
        return 1.95
       elif ltype=='Medium':
        return 1.535
       else:
        return 1.325
    df['luxury_category']=df['luxury_category'].apply(encode_luxury)
    
    def encode_ap(atype):
      if atype=='Moderately Old':
        return 1.85
      elif atype=='New Property':
        return 1.35
      elif atype=='Old Property':
        return 2.20
      elif atype=='Relatively New':
        return 1.45
      else:
        return 1.33
    df['agePossession']=df['agePossession'].apply(encode_ap)
    
    def encode_ptype(ptype):
        if ptype=='flat':
            return 1.38
        else:
            return 4
    df['property_type'] = df['property_type'].apply(encode_ptype)
    
    return df

if st.button('Predict'):

    # form a dataframe
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']
    
   
    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)
    st.dataframe(one_df)
    #st.dataframe(one_df)
    one_df = transform_input(one_df)
    
    expected_df = pd.read_csv('X.csv')
    expected_columns = expected_df.columns.tolist()
    one_df= one_df.reindex(columns=expected_columns)
    # predict
    base_price = np.expm1(xgb_model.predict(one_df))[0]
    low = base_price - 0.22
    high = base_price + 0.22

    # display
    st.text("The price of the flat is between {} Cr and {} Cr".format(round(low,2),round(high,2)))
