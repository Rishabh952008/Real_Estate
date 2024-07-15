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

y_transformed = np.log1p(y)
columns_to_encode = ['property_type', 'balcony', 'furnishing_type', 'luxury_category', 'floor_category']

# Creating a column transformer for preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(), columns_to_encode),
        ('cat1',OneHotEncoder(drop='first',sparse_output=False),['sector','agePossession'])
    ], 
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor())
])

pipeline.fit(X,y_transformed)

with open('models/pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

with open('data/raw/df.pkl', 'wb') as file:
    pickle.dump(X, file)

