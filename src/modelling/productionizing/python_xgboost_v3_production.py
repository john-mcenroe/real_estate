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
input_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/final_test_predictions_xgboost_v3.csv'
df = pd.read_csv(input_path)

# Define target and features
target = 'sale_price'
features = [
    'beds', 'baths', 'myhome_floor_area_value', 'latitude', 'longitude',
    'nearby_properties_count_within_1km',
    'avg_sold_price_within_1km', 'median_sold_price_within_1km',
    'avg_asking_price_within_1km', 'median_asking_price_within_1km',
    'avg_price_delta_within_1km', 'median_price_delta_within_1km',
    'nearby_properties_count_within_3km',
    'avg_sold_price_within_3km', 'median_sold_price_within_3km',
    'avg_asking_price_within_3km', 'median_asking_price_within_3km',
    'avg_price_delta_within_3km', 'median_price_delta_within_3km',
    'nearby_properties_count_within_5km',
    'avg_sold_price_within_5km', 'median_sold_price_within_5km',
    'avg_asking_price_within_5km', 'median_asking_price_within_5km',
    'avg_price_delta_within_5km', 'median_price_delta_within_5km',
    'property_type', 'energy_rating'
]

# Ensure all required features are present
available_features = [f for f in features if f in df.columns]
if len(available_features) != len(features):
    missing_features = set(features) - set(available_features)
    print(f"Warning: The following features are missing from the dataset: {missing_features}")
    features = available_features

# Handle missing values and infinities
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + [target])

# Split the data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
numerical_cols = [col for col in [
    'beds', 'baths', 'size', 'latitude', 'longitude',
    'nearby_properties_count_within_1km',
    'avg_sold_price_within_1km', 'median_sold_price_within_1km',
    'avg_asking_price_within_1km', 'median_asking_price_within_1km',
    'avg_price_delta_within_1km', 'median_price_delta_within_1km',
    'nearby_properties_count_within_3km',
    'avg_sold_price_within_3km', 'median_sold_price_within_3km',
    'avg_asking_price_within_3km', 'median_asking_price_within_3km',
    'avg_price_delta_within_3km', 'median_price_delta_within_3km',
    'nearby_properties_count_within_5km',
    'avg_sold_price_within_5km', 'median_sold_price_within_5km',
    'avg_asking_price_within_5km', 'median_asking_price_within_5km',
    'avg_price_delta_within_5km', 'median_price_delta_within_5km'
] if col in features]
categorical_cols = [col for col in ['property_type', 'energy_rating'] if col in features]

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

# Add these lines for debugging
print(f"y_test shape: {y_test.shape}")
print(f"y_pred shape: {y_pred.shape}")
print(f"y_test sample: {y_test.head()}")
print(f"y_pred sample: {y_pred[:5]}")

# Check for NaN or infinite values
print(f"NaN in y_test: {np.isnan(y_test).any()}")
print(f"NaN in y_pred: {np.isnan(y_pred).any()}")
print(f"Inf in y_test: {np.isinf(y_test).any()}")
print(f"Inf in y_pred: {np.isinf(y_pred).any()}")

# Save the model
model_path = os.path.join(os.path.dirname(input_path), 'xgboost_model.joblib')
joblib.dump(model_pipeline, model_path)
print(f"Saved trained model to {model_path}")

# Function to make predictions
def predict_price(**kwargs):
    input_data = pd.DataFrame([kwargs])
    prediction = model_pipeline.predict(input_data)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    print("\nRunning example prediction:")
    example_price = predict_price(
        beds=3,
        baths=2,
        myhome_floor_area_value=120,
        latitude=53.3498,
        longitude=-6.2603,
        nearby_properties_count_within_1km=50,
        avg_sold_price_within_1km=400000,
        median_sold_price_within_1km=380000,
        avg_asking_price_within_1km=420000,
        median_asking_price_within_1km=400000,
        avg_price_delta_within_1km=20000,
        median_price_delta_within_1km=20000,
        nearby_properties_count_within_3km=150,
        avg_sold_price_within_3km=380000,
        median_sold_price_within_3km=360000,
        avg_asking_price_within_3km=400000,
        median_asking_price_within_3km=380000,
        avg_price_delta_within_3km=20000,
        median_price_delta_within_3km=20000,
        nearby_properties_count_within_5km=300,
        avg_sold_price_within_5km=360000,
        median_sold_price_within_5km=340000,
        avg_asking_price_within_5km=380000,
        median_asking_price_within_5km=360000,
        avg_price_delta_within_5km=20000,
        median_price_delta_within_5km=20000,
        property_type="House",
        energy_rating="B1"
    )
    print(f"Predicted price: €{example_price:,.2f}")
