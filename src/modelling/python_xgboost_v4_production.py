import pandas as pd
import numpy as np
import os
import warnings
import joblib
from datetime import datetime

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

# For hyperparameter tuning and cross-validation
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import uniform, randint

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

# Update the input file path
input_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/processed_data_v3_output.csv'

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

# Define target
target = 'sale_price'

# Explicitly define features
features = [
    'beds', 'baths', 'property_type', 'energy_rating',
    'myhome_floor_area_value', 'latitude', 'longitude',
    'bedCategory', 'bathCategory', 'propertyTypeCategory', 'berCategory', 'sizeCategory',
    'size', 'nearby_properties_count_within_1km', 'nearby_properties_count_within_3km',
    'nearby_properties_count_within_5km',
    '30d_1km_num_properties_sold', '90d_1km_median_sold_price', '90d_1km_avg_asking_price',
    '90d_1km_num_properties_sold', '90d_1km_avg_days_on_market', '90d_1km_median_price_per_sqm',
    '180d_1km_median_sold_price', '180d_1km_avg_asking_price', '180d_1km_num_properties_sold',
    '180d_1km_avg_days_on_market', '180d_1km_median_price_per_sqm',
    '1km_ber_dist_C', '1km_ber_dist_D', '1km_ber_dist_B', '1km_ber_dist_F', '1km_ber_dist_A',
    '1km_property_type_dist_Apartment', '1km_property_type_dist_Semi-D', '1km_property_type_dist_Terrace',
    '1km_avg_property_size', '1km_median_beds', '1km_median_baths', '1km_price_to_income_ratio',
    '1km_price_growth_rate',
    '30d_3km_num_properties_sold', '90d_3km_median_sold_price', '90d_3km_avg_asking_price',
    '90d_3km_num_properties_sold', '90d_3km_avg_days_on_market', '90d_3km_median_price_per_sqm',
    '180d_3km_median_sold_price', '180d_3km_avg_asking_price', '180d_3km_num_properties_sold',
    '180d_3km_avg_days_on_market', '180d_3km_median_price_per_sqm',
    '3km_ber_dist_C', '3km_ber_dist_D', '3km_ber_dist_B', '3km_ber_dist_E', '3km_ber_dist_A',
    '3km_ber_dist_F', '3km_ber_dist_Unknown',
    '3km_property_type_dist_Semi-D', '3km_property_type_dist_Apartment', '3km_property_type_dist_Terrace',
    '3km_property_type_dist_Detached', '3km_property_type_dist_End of Terrace', '3km_property_type_dist_Bungalow',
    '3km_property_type_dist_Duplex',
    '3km_avg_property_size', '3km_median_beds', '3km_median_baths', '3km_price_to_income_ratio',
    '3km_price_growth_rate',
    '30d_5km_num_properties_sold', '90d_5km_median_sold_price', '90d_5km_avg_asking_price',
    '90d_5km_num_properties_sold', '90d_5km_avg_days_on_market', '90d_5km_median_price_per_sqm',
    '180d_5km_median_sold_price', '180d_5km_avg_asking_price', '180d_5km_num_properties_sold',
    '180d_5km_avg_days_on_market', '180d_5km_median_price_per_sqm',
    '5km_ber_dist_C', '5km_ber_dist_D', '5km_ber_dist_B', '5km_ber_dist_A', '5km_ber_dist_E',
    '5km_ber_dist_Unknown', '5km_ber_dist_F', '5km_ber_dist_G',
    '5km_property_type_dist_Semi-D', '5km_property_type_dist_Apartment', '5km_property_type_dist_Terrace',
    '5km_property_type_dist_Detached', '5km_property_type_dist_End of Terrace', '5km_property_type_dist_Duplex',
    '5km_property_type_dist_Bungalow', '5km_property_type_dist_Townhouse', '5km_property_type_dist_Site',
    '5km_avg_property_size', '5km_median_beds', '5km_median_baths', '5km_price_to_income_ratio',
    '5km_price_growth_rate',
    '1km_ber_dist_E', '1km_ber_dist_G', '1km_ber_dist_Unknown',
    '1km_property_type_dist_End of Terrace', '3km_ber_dist_G',
    '1km_property_type_dist_Duplex', '1km_property_type_dist_Studio', '1km_property_type_dist_Townhouse',
    '1km_property_type_dist_Bungalow', '1km_property_type_dist_Detached',
    '3km_property_type_dist_Townhouse', '3km_property_type_dist_Studio', '3km_property_type_dist_Site',
    '5km_property_type_dist_Studio', '5km_property_type_dist_Houses', '3km_property_type_dist_Houses',
    '1km_property_type_dist_Site', '1km_property_type_dist_Houses'
]

# Add these new features to your features list
features += [
    'days_on_market',
    'price_change_percentage',
    'price_per_bedroom',
    'price_per_bathroom',
    'total_room_count',
    'bedroom_to_bathroom_ratio',
    'property_age',
    'distance_to_city_center',
    'season_of_sale'
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
    ('regressor', XGBRegressor(random_state=42, n_jobs=-1))
])

# =========================================
# 10. Hyperparameter Tuning
# =========================================
# Define the parameter space
param_distributions = {
    'regressor__n_estimators': randint(100, 1000),
    'regressor__max_depth': randint(3, 10),
    'regressor__learning_rate': uniform(0.01, 0.3),
    'regressor__subsample': uniform(0.6, 0.4),
    'regressor__colsample_bytree': uniform(0.6, 0.4),
    'regressor__min_child_weight': randint(1, 10),
    'regressor__gamma': uniform(0, 5)
}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    model_pipeline,
    param_distributions=param_distributions,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# Fit the random search
print("Starting hyperparameter tuning...")
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score: {:.2f}".format(-random_search.best_score_))

# Update the model pipeline with the best parameters
model_pipeline = random_search.best_estimator_

# =========================================
# 11. Cross-Validation
# =========================================
print("Performing cross-validation...")
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

print("Cross-validation RMSE scores:", cv_rmse_scores)
print("Mean RMSE: {:.2f}".format(cv_rmse_scores.mean()))
print("Standard deviation of RMSE: {:.2f}".format(cv_rmse_scores.std()))

# =========================================
# 12. Train the Final Model
# =========================================
print("Training the final model...")
model_pipeline.fit(X_train, y_train)

# =========================================
# 13. Make Predictions on the Test Set
# =========================================
y_pred = model_pipeline.predict(X_test)

# =========================================
# 14. Evaluate the Model
# =========================================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nXGBoost Regressor Performance on Test Set:")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# =========================================
# 15. Feature Importance Analysis
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
# 16. Visualize Feature Importances
# =========================================
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# =========================================
# 17. Save the Trained Model
# =========================================

# Define the path to save the model
model_path = os.path.join(os.path.dirname(input_path), 'xgboost_model.joblib')

# Save the entire pipeline (including preprocessor and model)
joblib.dump(model_pipeline, model_path)
print(f"\nSaved trained model to {model_path}")

# =========================================
# 18. Define the `predict_and_compare` Function
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
# 19. Example Usage of `predict_and_compare`
# =========================================

# Example: Predict and compare for the first sample in the test set
predict_and_compare(sample_index=0)

# =========================================
# 20. Save the Final Test Dataset with Predictions as CSV
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
# 21. Simple Validation (Optional)
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
# 22. Optional: Learning Curves
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

# Add these constants at the top of your script
DUBLIN_CENTER_LAT = 53.3498  # Latitude of Dublin city center
DUBLIN_CENTER_LON = -6.2603  # Longitude of Dublin city center




