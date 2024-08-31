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
import logging


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        raise

def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess the dataset for machine learning.

    Args:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - tuple: X (features), y (target).
    """
    try:
        # Filter out rows with null floor_category
        df = df[~df['floor_category'].isnull()]

        # Replace furnishing_type values
        df['furnishing_type'] = df['furnishing_type'].replace({0.0: 'unfurnished', 1.0: 'semifurnished', 2.0: 'furnished'})

        # Filter out specific conditions for property_type and price
        df = df[~((df['property_type'] == 'flat') & (df['price'] == 14.00))]

        # Split features (X) and target (y)
        X = df.drop(columns=['price'])
        y = df['price']


        # Transform target y
        y_transformed = np.log1p(y)
        
        logging.info("Data preprocessing completed successfully")

        return X, y_transformed
    except KeyError as e:
        logging.error(f"KeyError: {e}. Check column names in the DataFrame.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {str(e)}")
        raise


def build_pipeline() -> Pipeline:
    """
    Build a machine learning pipeline.

    Returns:
    - Pipeline: Constructed machine learning pipeline.
    """
    try:
        # Define columns to encode
        columns_to_encode = ['property_type', 'balcony', 'furnishing_type', 'luxury_category', 'floor_category']

        # Create a column transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
                ('cat', OrdinalEncoder(), columns_to_encode),
                ('cat1', OneHotEncoder(drop='first', sparse_output=False), ['sector', 'agePossession'])
            ],
            remainder='passthrough'
        )

        # Construct the pipeline with preprocessor and XGBRegressor
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor())
        ])

        logging.info("Pipeline construction completed successfully")

        return pipeline
    except Exception as e:
        logging.error(f"An error occurred during pipeline construction: {str(e)}")
        raise

def save_model(pipeline: Pipeline, model_file: str, data_file: str,X: pd.DataFrame):
    """
    Save the trained pipeline and input data.

    Args:
    - pipeline (Pipeline): Trained machine learning pipeline.
    - model_file (str): File path to save the trained model.
    - data_file (str): File path to save the input data.
    """
    try:
        with open(model_file, 'wb') as file:
            pickle.dump(pipeline, file)
        logging.info(f"Trained pipeline saved successfully to {model_file}")

        with open(data_file, 'wb') as file:
            pickle.dump(X, file)
        logging.info(f"Input data saved successfully to {data_file}")

    except Exception as e:
        logging.error(f"An error occurred during model or data saving: {str(e)}")
        raise


def main():
    try:
        # Load data
        file_path = 'prepared_data/post_feature_selection.csv'
        df = load_data(file_path)

        # Preprocess data
        X, y_transformed = preprocess_data(df)
        
        X.to_csv('data/processed/X.csv',index=False)
        y_transformed.to_csv('data/processed/y_transformed.csv',index=False)

        # Build pipeline
        pipeline = build_pipeline()

        # Train pipeline
        pipeline.fit(X, y_transformed)

        # Save model and data
        model_file = 'models/pipeline.pkl'
        data_file = 'data/raw/df.pkl'
        save_model(pipeline, model_file, data_file,X)

        logging.info("Pipeline training and saving completed successfully")

    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    main()


# df = pd.read_csv('prepared_data/post_feature_selection.csv')

# df = df[~(df['floor_category'].isnull())]

# # 0 -> unfurnished
# # 1 -> semifurnished
# # 2 -> furnished
# df['furnishing_type'] = df['furnishing_type'].replace({0.0:'unfurnished',1.0:'semifurnished',2.0:'furnished'})

# df = df[~((df['property_type']=='flat') & (df['price']==14.00))]

# X = df.drop(columns=['price'])
# y = df['price']

# y_transformed = np.log1p(y)
# columns_to_encode = ['property_type', 'balcony', 'furnishing_type', 'luxury_category', 'floor_category']

# # Creating a column transformer for preprocessing

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
#         ('cat', OrdinalEncoder(), columns_to_encode),
#         ('cat1',OneHotEncoder(drop='first',sparse_output=False),['sector','agePossession'])
#     ], 
#     remainder='passthrough'
# )

# pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', XGBRegressor())
# ])

# pipeline.fit(X,y_transformed)


# with open('models/pipeline.pkl', 'wb') as file:
#     pickle.dump(pipeline, file)

# with open('data/raw/df.pkl', 'wb') as file:
#     pickle.dump(X, file)

