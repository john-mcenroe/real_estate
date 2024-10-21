import pandas as pd
import numpy as np
import os
import warnings
import joblib

# For preprocessing and model training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import XGBoost Regressor
from xgboost import XGBRegressor

# For handling missing values
from sklearn.impute import SimpleImputer

# For evaluation
from sklearn.metrics import mean_squared_error, r2_score

# For feature importance visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For version checking
import sklearn
from packaging import version

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =========================================
# 0. Version and Environment Checks
# =========================================

# Check scikit-learn version
sklearn_version = sklearn.__version__
print(f"scikit-learn version: {sklearn_version}")

# =========================================
# 1. Load the Data
# =========================================

# Define the input CSV path
input_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/full_run_predictions_xgboost_v3.csv'

# Check if file exists and its size
if not os.path.exists(input_path):
    raise FileNotFoundError(f"The input file does not exist at the specified path: {input_path}")

file_size = os.path.getsize(input_path)
print(f"File exists: {os.path.exists(input_path)}")
print(f"File size: {file_size} bytes")

# Load the DataFrame
df = pd.read_csv(input_path)

# =========================================
# 2. Initial Data Inspection
# =========================================
print(f"Initial number of rows: {len(df)}")
print(f"Initial number of columns: {len(df.columns)}")
print("\nFirst few rows of the dataframe:")
print(df.head())
print("\nDataframe info:")
print(df.info())

# =========================================
# 3. Define Target and Features
# =========================================

# Define target and features as per Script #2
target = 'sale_price'
features = [
    'beds', 'baths', 'myhome_floor_area_value', 'latitude', 'longitude',
    'energy_rating_numeric', 'bedCategory', 'bathCategory', 'propertyTypeCategory', 'berCategory', 'sizeCategory',
    'nearby_properties_count_within_1km',
    'avg_sold_price_within_1km', 'median_sold_price_within_1km',
    'avg_asking_price_within_1km', 'median_asking_price_within_1km',
    'avg_price_delta_within_1km', 'median_price_delta_within_1km',
    'avg_price_per_sqm_within_1km', 'median_price_per_sqm_within_1km',
    'avg_bedrooms_within_1km', 'avg_bathrooms_within_1km',
    'nearby_properties_count_within_3km',
    'avg_sold_price_within_3km', 'median_sold_price_within_3km',
    'avg_asking_price_within_3km', 'median_asking_price_within_3km',
    'avg_price_delta_within_3km', 'median_price_delta_within_3km',
    'avg_price_per_sqm_within_3km', 'median_price_per_sqm_within_3km',
    'avg_bedrooms_within_3km', 'avg_bathrooms_within_3km',
    'nearby_properties_count_within_5km',
    'avg_sold_price_within_5km', 'median_sold_price_within_5km',
    'avg_asking_price_within_5km', 'median_asking_price_within_5km',
    'avg_price_delta_within_5km', 'median_price_delta_within_5km',
    'avg_price_per_sqm_within_5km', 'median_price_per_sqm_within_5km',
    'avg_bedrooms_within_5km', 'avg_bathrooms_within_5km',
    'property_type', 'energy_rating'
]

# Ensure all required features are present
available_features = [f for f in features if f in df.columns]
if len(available_features) != len(features):
    missing_features = set(features) - set(available_features)
    print(f"\nWarning: The following features are missing from the dataset: {missing_features}")
    features = available_features

print(f"\nNumber of features: {len(features)}")
print(f"Target column: {target}")

# =========================================
# 4. Data Cleaning
# =========================================

# Handle missing values and infinities
df = df.replace([np.inf, -np.inf], np.nan)
print(f"\nNumber of rows after replacing inf: {len(df)}")

# Print missing value information
print("\nMissing values in each column:")
print(df[features + [target]].isnull().sum())

# =========================================
# 5. Separate Features and Target
# =========================================
X = df[features]
y = df[target]

# =========================================
# 6. Split the Data into Training and Testing Sets
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Test target shape:", y_test.shape)

# =========================================
# 7. Identify Numerical and Categorical Columns
# =========================================
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# =========================================
# 8. Preprocessing Pipelines
# =========================================

# Numerical transformer: Impute missing values with median and scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer: Impute missing values with 'Unknown' and apply One-Hot Encoding
if version.parse(sklearn_version) >= version.parse("1.2"):
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', onehot_encoder)  # Ensure this step is named 'onehot'
])

# Combine numerical and categorical transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# =========================================
# 9. Create the Modeling Pipeline with XGBoost
# =========================================
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'  # Specify the objective for regression
    ))
])

# =========================================
# 10. Train the Model
# =========================================
try:
    model_pipeline.fit(X_train, y_train)
    print("\nModel training completed.")
except ValueError as e:
    print("\nError during model training:", e)
    raise

# =========================================
# 11. Make Predictions on the Test Set
# =========================================
y_pred = model_pipeline.predict(X_test)

# =========================================
# 12. Evaluate the Model
# =========================================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nXGBoost Regressor Performance on Test Set:")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# =========================================
# 13. Feature Importance Analysis
# =========================================
def get_feature_names(column_transformer):
    feature_names = []
    for name, transformer, features in column_transformer.transformers_:
        if name != 'remainder':
            if hasattr(transformer, 'get_feature_names_out'):
                if isinstance(features, slice):
                    features = column_transformer._feature_names_in[features]
                feature_names.extend(transformer.get_feature_names_out(features).tolist())
            elif hasattr(transformer, 'named_steps'):
                # This is a pipeline
                if 'onehot' in transformer.named_steps:
                    feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out(features).tolist())
                else:
                    feature_names.extend(features)
            else:
                feature_names.extend(features)
    return feature_names

feature_names = get_feature_names(model_pipeline.named_steps['preprocessor'])

print(f"Number of feature names: {len(feature_names)}")
print(f"Number of feature importances: {len(model_pipeline.named_steps['regressor'].feature_importances_)}")

if len(feature_names) != len(model_pipeline.named_steps['regressor'].feature_importances_):
    print("Warning: Mismatch between number of feature names and feature importances.")
    print("Using range-based feature names instead.")
    feature_names = [f'feature_{i}' for i in range(len(model_pipeline.named_steps['regressor'].feature_importances_))]

feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model_pipeline.named_steps['regressor'].feature_importances_
})

feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

top_n = 20
top_features = feature_importances.head(top_n)

print(f"\nTop {top_n} Feature Importances:")
print(top_features)

# =========================================
# 14. Visualize Feature Importances
# =========================================
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# =========================================
# 15. Save the Trained Model
# =========================================

# Define the path to save the model
model_path = os.path.join(os.path.dirname(input_path), 'xgboost_model.joblib')

# Save the entire pipeline (including preprocessor and model)
joblib.dump(model_pipeline, model_path)
print(f"\nSaved trained model to {model_path}")

# =========================================
# 16. Define the `predict_and_compare` Function
# =========================================

def predict_and_compare(sample_index=None, unique_id=None):
    """
    Predicts the sold price for a specific row and compares it with the actual price.
    
    Parameters:
    - sample_index (int): The index of the row in X_test.
    - unique_id (Any): The unique identifier of the row (if applicable).
    
    Note: Provide either sample_index or unique_id.
    """
    if sample_index is not None:
        try:
            sample = X_test.iloc[[sample_index]]  # Select as DataFrame
            actual_price = y_test.iloc[sample_index]
            
            # Access preserved details if available
            preserved_details = df.iloc[X_test.index[sample_index]]
            
            print(f"\nSelected Row Index: {X_test.index[sample_index]}")
        except IndexError:
            print("Error: Sample index out of range.")
            return
    elif unique_id is not None:
        # Assuming there is an 'ID' column; adjust accordingly
        if 'ID' not in df.columns:
            print("Error: 'ID' column not found in the dataset.")
            return
        sample = X_test[X_test.index == unique_id]
        if sample.empty:
            print(f"Error: No row found with ID = {unique_id}.")
            return
        actual_price = y_test[X_test.index == unique_id].iloc[0]
        
        # Access preserved details if available
        preserved_details = df.loc[unique_id]
        
        print(f"\nSelected Row ID: {unique_id}")
    else:
        print("Error: Please provide either sample_index or unique_id.")
        return
    
    # Display Preserved Property Details if available
    if 'beds' in preserved_details and 'baths' in preserved_details:
        print(f"Beds: {preserved_details['beds']}")
        print(f"Baths: {preserved_details['baths']}")
    if 'myhome_floor_area_value' in preserved_details:
        print(f"Floor Area Value: {preserved_details['myhome_floor_area_value']}")
    if 'latitude' in preserved_details and 'longitude' in preserved_details:
        print(f"Latitude: {preserved_details['latitude']}")
        print(f"Longitude: {preserved_details['longitude']}")
    if 'property_type' in preserved_details:
        print(f"Property Type: {preserved_details['property_type']}")
    if 'energy_rating' in preserved_details:
        print(f"Energy Rating: {preserved_details['energy_rating']}")
    
    print(f"Actual Sold Price: €{actual_price:,.2f}")
    
    # Predict
    try:
        predicted_price = model_pipeline.predict(sample)[0]
        print(f"Predicted Sold Price: €{predicted_price:,.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Compare
    difference = predicted_price - actual_price
    try:
        percentage_error = (difference / actual_price) * 100
    except ZeroDivisionError:
        percentage_error = np.nan  # Handle division by zero if actual_price is 0
    
    print(f"Difference: €{difference:,.2f}")
    if not np.isnan(percentage_error):
        print(f"Percentage Error: {percentage_error:.2f}%")
    else:
        print("Percentage Error: Undefined (Actual Sold Price is 0)")

# =========================================
# 17. Example Usage of `predict_and_compare`
# =========================================

# Example: Predict and compare for the first sample in the test set
predict_and_compare(sample_index=0)

# =========================================
# 18. Save the Final Test Dataset with Predictions as CSV
# =========================================

# Create a DataFrame for test set predictions
test_predictions = pd.DataFrame({
    'Actual_Sale_Price': y_test,
    'Predicted_Sale_Price': y_pred
}, index=X_test.index)

# Combine with original test features if needed
# Here, we include key features for reference
key_features = ['beds', 'baths', 'myhome_floor_area_value', 'property_type', 'energy_rating']
available_key_features = [feat for feat in key_features if feat in df.columns]
test_predictions = test_predictions.join(df.loc[X_test.index, available_key_features])

# Calculate Difference and Percentage Difference
test_predictions['Difference (€)'] = test_predictions['Predicted_Sale_Price'] - test_predictions['Actual_Sale_Price']
test_predictions['Percentage_Difference (%)'] = (test_predictions['Difference (€)'] / test_predictions['Actual_Sale_Price']) * 100

print("\nFinal Test Dataset with Predictions:")
print(test_predictions.head())

# Save the predictions to CSV
output_filename = 'final_test_predictions_xgboost.csv'
output_path = os.path.join(os.path.dirname(input_path), output_filename)
test_predictions.to_csv(output_path, index=False)
print(f"\nSaved final test predictions to {output_path}")

# =========================================
# 19. Simple Validation (Optional)
# =========================================

# Further split the training data into training and validation sets
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Retrain the model on the final training set
model_pipeline.fit(X_train_final, y_train_final)

# Predict on the validation set
y_val_pred = model_pipeline.predict(X_val)

# Calculate RMSE and R² for the validation set
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

print("\nValidation Set Performance:")
print(f"Root Mean Squared Error (RMSE): {val_rmse:,.2f}")
print(f"R² Score: {val_r2:.4f}")

# =========================================
# 20. Optional: Learning Curves
# =========================================
# Uncomment the following section if you wish to plot learning curves

'''
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training RMSE')
plt.plot(train_sizes, test_scores_mean, label='Validation RMSE')
plt.xlabel('Training Set Size')
plt.ylabel('Root Mean Squared Error')
plt.title('Learning Curves')
plt.legend()
plt.show()
'''

# =========================================
# End of Script
# =========================================