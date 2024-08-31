import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import yaml
import json
from dvclive import Live

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

def build_pipeline(n_estimators,max_depth) -> Pipeline:
    """
    Build a machine learning pipeline.

    Args:
    - model_params (dict): Parameters for the model.

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

        # Select model based on parameters
     
        model = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth)
   

        # Construct the pipeline with preprocessor and selected model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        logging.info("Pipeline construction completed successfully")

        return pipeline
    except Exception as e:
        logging.error(f"An error occurred during pipeline construction: {str(e)}")
        raise

def save_model(pipeline: Pipeline, model_file: str):
    """
    Save the trained pipeline.

    Args:
    - pipeline (Pipeline): Trained machine learning pipeline.
    - model_file (str): File path to save the trained model.
    """
    try:
        with open(model_file, 'wb') as file:
            pickle.dump(pipeline, file)
        logging.info(f"Trained pipeline saved successfully to {model_file}")

    except Exception as e:
        logging.error(f"An error occurred during model saving: {str(e)}")
        raise

def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,param):
    """
    Evaluate the trained pipeline.

    Args:
    - pipeline (Pipeline): Trained machine learning pipeline.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series): Test target.
    """
    try:
        # Predict
        y_pred = pipeline.predict(X_test)

        # Calculate evaluation metrics
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        with Live(save_dvc_exp=True) as live:
            live.log_metric('r2_score',r2)
            live.log_metric('mean_absolute_error',mae)
            live.log_metric('rmse',rmse)
             
            live.log_param('n_estimators',param['n_estimators'])
            live.log_param('max_depth',param['max_depth'])
            live.log_param('test_size',param['test_size'])
        
        # metrics_dict = {
        #     'r2_score':r2,
        #     'mean_absolute_error':mae,
        #     'rmse':rmse
        # }
        # with open('metrics.json','w') as file:
        #     json.dump(metrics_dict,file,indent=4)

        logging.info(f"Model evaluation metrics:\nR2 Score: {r2}\nMAE: {mae}\nRMSE: {rmse}")

    except Exception as e:
        logging.error(f"An error occurred during model evaluation: {str(e)}")
        raise

def main():
    try:
        # Load data
        X = load_data('data/processed/X.csv')

        # Load target variable y_transformed
        y_transformed = pd.read_csv('data/processed/y_transformed.csv')

        # Load model parameters from YAML
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)

        # List of models to try out
        param = params['model_evaluation']
        
            # Build pipeline
        pipeline = build_pipeline(n_estimators=param['n_estimators'],max_depth=param['max_depth'])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=param['test_size'], random_state=42)

            # Train pipeline
        pipeline.fit(X_train, y_train)

            # Evaluate model
        evaluate_model(pipeline, X_test, y_test,param=param)

            # Save model
        # model_file = f'models/rf_model.pkl'
        # save_model(pipeline, model_file)

        logging.info("Pipeline training, evaluation, and saving completed successfully")

    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    main()
