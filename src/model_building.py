import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pickle 

df = pd.read_csv('prepared_data/post_feature_selection.csv')

df = df[~(df['floor_category'].isnull())]

# 0 -> unfurnished
# 1 -> semifurnished
# 2 -> furnished
df['furnishing_type'] = df['furnishing_type'].replace({0.0:'unfurnished',1.0:'semifurnished',2.0:'furnished'})

df = df[~((df['property_type']=='flat') & (df['price']==14.00))]

X = df.drop(columns=['price'])
y = df['price']

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

X = transform_input(X)
    

# Applying the log1p transformation to the target variable
y_transformed = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X,y , 
                                   random_state=104,  
                                   test_size=0.30,  
                                   shuffle=True) 

xgb_model = XGBRegressor()

xgb_model.fit(X_train,y_train)

with open('models/xgb_model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)



