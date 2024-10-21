import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load the data
input_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/preprocess_output_test.csv'
df = pd.read_csv(input_path)

# Define target and features
target = 'Sold Price'
features = ['Address', 'Beds', 'Baths', 'Property Type', 'Energy Rating']

# Ensure all required features are present
for feature in features:
    if feature not in df.columns:
        raise ValueError(f"Required feature '{feature}' not found in the dataset.")

# Handle missing values and infinities
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + [target])

# Split the data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
numerical_cols = ['Beds', 'Baths']
categorical_cols = ['Address', 'Property Type', 'Energy Rating']

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

# Create the model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions and calculate R^2 score
y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Save the model
model_path = os.path.join(os.path.dirname(input_path), 'xgboost_model.joblib')
joblib.dump(model_pipeline, model_path)
print(f"Saved trained model to {model_path}")

# Function to make predictions
def predict_price(**kwargs):
    # Get the expected feature names
    expected_features = ['Address', 'Beds', 'Baths', 'Property Type', 'Energy Rating']
    
    # Create a DataFrame with all expected features
    input_data = pd.DataFrame(columns=expected_features)
    
    # Fill in provided values, leaving others as NaN
    for feature in expected_features:
        if feature in kwargs:
            input_data[feature] = [kwargs[feature]]
        else:
            print(f"Warning: '{feature}' not provided. It will be handled by the preprocessing step.")
    
    # Preprocess the input data
    preprocessor = model_pipeline.named_steps['preprocessor']
    preprocessed_data = preprocessor.transform(input_data)
    
    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()
    
    # Create a DataFrame with preprocessed data and feature names
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names)
    
    # Get the XGBoost model from the pipeline
    xgb_model = model_pipeline.named_steps['regressor']
    
    # Make prediction using the preprocessed data
    prediction = xgb_model.predict(preprocessed_df)
    
    return prediction[0]

# Example usage with the provided address
print("\nExample prediction:")
example_price = predict_price(
    Address="Cornerways, Grove Ave, Blackrock, Dublin",
    Beds=3,
    Baths=3,
    Property_Type="House",
    Energy_Rating="B1"
)
print(f"Predicted price: €{example_price:,.2f}")

# Add this to ensure the example runs when the script is executed
if __name__ == "__main__":
    print("\nRunning example prediction:")
    example_price = predict_price(
        Address="Cornerways, Grove Ave, Blackrock, Dublin",
        Beds=3,
        Baths=3,
        Property_Type="House",
        Energy_Rating="B1"
    )
    print(f"Predicted price: €{example_price:,.2f}")
