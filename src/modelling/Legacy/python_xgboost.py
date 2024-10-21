import pandas as pd
import numpy as np

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

# For handling file paths
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Check scikit-learn version
sklearn_version = sklearn.__version__
print(f"scikit-learn version: {sklearn_version}")

# =========================================
# 1. Load the Data
# =========================================

# Define the input CSV path
input_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/preprocess_output_test.csv'

# Load the DataFrame
df_with_metrics = pd.read_csv(input_path)

# Create a copy of the DataFrame
df = df_with_metrics.copy()

# =========================================
# 2. Initial Data Inspection
# =========================================
print("Initial Data Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

# =========================================
# 3. Preserve Relevant Columns Before Dropping
# =========================================

# Define the columns you want to preserve, including 'MyHome_URL'
columns_to_preserve = {
    'Address': ['Address', 'MyHome_Address'],
    'Beds': ['Beds', 'MyHome_Beds'],
    'Baths': ['Baths', 'MyHome_Baths'],
    'Sold Price': ['Sold Price', 'MyHome_Sold_Price'],
    'Floor Area Value': ['MyHome_Floor_Area_Value'],  # Adjusted based on your data
    'Latitude': ['MyHome_Latitude'],
    'Longitude': ['MyHome_Longitude'],
    'URL': ['URL'],
    'MyHome_URL': ['MyHome_Link']  # Added MyHome_URL
}

# Initialize an empty DataFrame to store preserved columns
preserved_df = pd.DataFrame(index=df.index)

# Extract each column if it exists
for key, possible_cols in columns_to_preserve.items():
    for col in possible_cols:
        if col in df.columns:
            preserved_df[key] = df[col]
            break
    else:
        # If none of the possible columns are found, assign 'N/A' or appropriate default
        preserved_df[key] = 'N/A'

print("\nPreserved Columns:")
print(preserved_df.head())

# =========================================

# =========================================
# 4. Drop the 'price_per_square_meter' Column if present
# =========================================
if 'price_per_square_meter' in df.columns:
    df = df.drop('price_per_square_meter', axis=1)
    print("\nDropped 'price_per_square_meter' column.")

# =========================================
# 5. Drop Irrelevant, Leakage Columns, and Ensure Local Property Tax is Removed
# =========================================
columns_to_drop = [
    'Address', 'URL', 'MyHome_Address', 'MyHome_Link',
    'Agency Name', 'Agency Contact', 'Sold Asking Price',
    'Asking Price',  # Dropped "Asking Price" from inputs
    'First List Date',
    'Sold Date',      # Dropped "Sold Date" from inputs
    'Local Property Tax'  # Corrected column name
    # 'Sold Price' is intentionally omitted to be used as target
]

# Drop columns if they exist in the dataframe
existing_drop_columns = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=existing_drop_columns, axis=1)
print(f"\nDropped columns: {existing_drop_columns}")

# =========================================
# 6. Define the Target Variable
# =========================================
target = 'Sold Price'  # Updated target variable

if target not in df.columns:
    raise ValueError(f"Target variable '{target}' not found in the dataset.")

# =========================================
# 7. Separate Features and Target
# =========================================
X = df.drop(target, axis=1)
y = df[target]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# =========================================
# 8. Further Drop Columns that May Cause Leakage or Are Irrelevant
# =========================================
leakage_columns = [
    'transaction_volume_num_sold_within_3km_90days',
    'transaction_volume_avg_monthly_transactions_within_3km'
]

existing_leakage_columns = [col for col in leakage_columns if col in X.columns]
X = X.drop(columns=existing_leakage_columns, axis=1)
print(f"\nDropped leakage columns: {existing_leakage_columns}")

# =========================================
# 9. Identify Numerical and Categorical Columns
# =========================================
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# =========================================
# 10. Handle High Cardinality Columns (Optional)
# =========================================
if 'Eircode' in categorical_cols:
    X = X.drop('Eircode', axis=1)
    categorical_cols.remove('Eircode')
    print("\nDropped 'Eircode' due to high cardinality.")

# =========================================
# 11. Ensure Numerical Columns are Numeric
# =========================================
for col in numerical_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# =========================================
# 12. Ensure Categorical Columns are of type 'str'
# =========================================
for col in categorical_cols:
    X[col] = X[col].astype(str)

# =========================================
# 13. Check for Remaining Missing Values
# =========================================
print("\nMissing values per column:")
print(X.isnull().sum())

# =========================================
# 14. Drop Columns with Excessive Missing Values
# =========================================
threshold = 0.5  # Drop columns with more than 50% missing values
missing_fraction = X.isnull().mean()
columns_to_drop_missing = missing_fraction[missing_fraction > threshold].index.tolist()
if columns_to_drop_missing:
    X = X.drop(columns=columns_to_drop_missing, axis=1)
    numerical_cols = [col for col in numerical_cols if col not in columns_to_drop_missing]
    categorical_cols = [col for col in categorical_cols if col not in columns_to_drop_missing]
    print(f"\nDropped columns with >50% missing values: {columns_to_drop_missing}")

# =========================================
# 15. Preprocessing Pipelines
# =========================================
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Adjust OneHotEncoder parameter based on scikit-learn version
if version.parse(sklearn_version) >= version.parse("1.2"):
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', onehot_encoder)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# =========================================
# 16. Split the Data into Training and Testing Sets
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# =========================================
# 17. Create the Modeling Pipeline with XGBoost
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
# 18. Train the Model
# =========================================
try:
    model_pipeline.fit(X_train, y_train)
    print("\nModel training completed.")
except ValueError as e:
    print("\nError during model training:", e)
    raise

# =========================================
# 19. Make Predictions on the Test Set
# =========================================
y_pred = model_pipeline.predict(X_test)

# =========================================
# 20. Evaluate the Model
# =========================================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nXGBoost Regressor Performance on Test Set:")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R² Score: {r2:.2f}")

# =========================================
# 21. Feature Importance Analysis
# =========================================
def get_feature_names(preprocessor, numerical_cols, categorical_cols):
    output_features = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            output_features.extend(columns)
        elif name == 'cat':
            onehot = transformer.named_steps['onehot']
            if hasattr(onehot, 'get_feature_names_out'):
                cat_features = onehot.get_feature_names_out(columns)
            else:
                cat_features = onehot.get_feature_names(columns)
            output_features.extend(cat_features)
    return output_features

feature_names = get_feature_names(preprocessor, numerical_cols, categorical_cols)

if len(feature_names) != len(model_pipeline.named_steps['regressor'].feature_importances_):
    raise ValueError("Mismatch between number of feature names and feature importances.")

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
# 22. Visualize Feature Importances
# =========================================
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# =========================================
# 23. Combine Preserved Test Data with Predictions
# =========================================

# Since we have split the data, align preserved_df with X_test
preserved_test = preserved_df.loc[X_test.index]

# Create a final DataFrame with preserved details, actual prices, and predictions
final_test_df = preserved_test.copy()

# Add Actual and Predicted Sold Prices
final_test_df['Actual Sold Price'] = y_test.values
final_test_df['Predicted Sold Price'] = y_pred

# =========================================
# **New Steps: Calculate 'Diff' and '% diff'**
# =========================================

# Calculate Difference and Percentage Difference
final_test_df['Diff'] = final_test_df['Predicted Sold Price'] - final_test_df['Actual Sold Price']
final_test_df['% diff'] = (final_test_df['Diff'] / final_test_df['Actual Sold Price']) * 100

# =========================================
# **Rearrange Columns in Desired Order**
# =========================================

# Define the desired column order
desired_columns = [
    'Address',
    'Beds',
    'Baths',
    'Floor Area Value',
    'MyHome_URL',
    'Sold Price',
    'Predicted Sold Price',
    'Diff',
    '% diff'
]

# Check if all desired columns are present
missing_columns = [col for col in desired_columns if col not in final_test_df.columns]
if missing_columns:
    print(f"Warning: The following desired columns are missing and will be filled with 'N/A': {missing_columns}")
    for col in missing_columns:
        final_test_df[col] = 'N/A'

# Reorder the columns
final_test_df = final_test_df[desired_columns]

print("\nFinal Test Dataset with Preserved Details and Predictions:")
print(final_test_df.head())

# =========================================
# 24. Save the Final Test Dataset with Predictions as CSV
# =========================================

# Define the output CSV path in the same directory as the input CSV
output_filename = 'final_test_predictions_xgboost.csv'
output_path = os.path.join(os.path.dirname(input_path), output_filename)

# Save the DataFrame to CSV
final_test_df.to_csv(output_path, index=False)

print(f"\nSaved final test predictions to {output_path}")

# =========================================
# 25. Define the `predict_and_compare` Function
# =========================================

# Function to predict and compare a specific row
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
            
            # Access preserved details from final_test_df
            address = final_test_df.iloc[sample_index]['Address']
            beds = final_test_df.iloc[sample_index]['Beds']
            baths = final_test_df.iloc[sample_index]['Baths']
            floor_area = final_test_df.iloc[sample_index]['Floor Area Value']
            myhome_url = final_test_df.iloc[sample_index]['MyHome_URL']
            sold_price = final_test_df.iloc[sample_index]['Sold Price']
            
            print(f"\nSelected Row Index: {X_test.index[sample_index]}")
        except IndexError:
            print("Error: Sample index out of range.")
            return
        except KeyError as e:
            print(f"Error: Missing column in final_test_df - {e}")
            return
    elif unique_id is not None:
        if 'ID' not in final_test_df.columns:
            print("Error: 'ID' column not found in the dataset.")
            return
        sample = X_test[X_test['ID'] == unique_id]
        if sample.empty:
            print(f"Error: No row found with ID = {unique_id}.")
            return
        sample = sample  # Already a DataFrame
        actual_price = y_test[X_test['ID'] == unique_id].iloc[0]
        
        # Access preserved details from final_test_df
        address = final_test_df[final_test_df['ID'] == unique_id]['Address'].iloc[0]
        beds = final_test_df[final_test_df['ID'] == unique_id]['Beds'].iloc[0]
        baths = final_test_df[final_test_df['ID'] == unique_id]['Baths'].iloc[0]
        floor_area = final_test_df[final_test_df['ID'] == unique_id]['Floor Area Value'].iloc[0]
        myhome_url = final_test_df[final_test_df['ID'] == unique_id]['MyHome_URL'].iloc[0]
        sold_price = final_test_df[final_test_df['ID'] == unique_id]['Sold Price'].iloc[0]
        
        print(f"\nSelected Row ID: {unique_id}")
    else:
        print("Error: Please provide either sample_index or unique_id.")
        return
    
    # Display Preserved Property Details
    print(f"Address: {address}")
    print(f"Beds: {beds}")
    print(f"Baths: {baths}")
    print(f"Floor Area Value: {floor_area}")
    print(f"MyHOME URL: {myhome_url}")
    print(f"Sold Price: {sold_price:,.2f}")
    
    print(f"Actual Sold Price: {actual_price:,.2f}")
    
    # Predict
    try:
        predicted_price = model_pipeline.predict(sample)[0]
        print(f"Predicted Sold Price: {predicted_price:,.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Compare
    difference = predicted_price - actual_price
    try:
        percentage_error = (difference / actual_price) * 100
    except ZeroDivisionError:
        percentage_error = np.nan  # Handle division by zero if actual_price is 0
    
    print(f"Difference: {difference:,.2f}")
    if not np.isnan(percentage_error):
        print(f"Percentage Error: {percentage_error:.2f}%")
    else:
        print("Percentage Error: Undefined (Actual Sold Price is 0)")

# =========================================
# 26. Example Usage of `predict_and_compare`
# =========================================

# Example: Predict and compare for the first sample in the test set
predict_and_compare(sample_index=0)

# =========================================
# 27. Save the Trained Model
# =========================================
import joblib

# Define the path to save the model
model_path = os.path.join(os.path.dirname(input_path), 'xgboost_model.joblib')

# Save the entire pipeline (including preprocessor and model)
joblib.dump(model_pipeline, model_path)
print(f"\nSaved trained model to {model_path}")

# =========================================
# 28. Simple Validation
# =========================================
from sklearn.model_selection import train_test_split

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model on the training set
model_pipeline.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = model_pipeline.predict(X_val)

# Calculate RMSE for the validation set
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("Validation RMSE:", val_rmse)

# =========================================
# 29. Learning Curves (Optional)
# =========================================
# Remove this section entirely or comment it out if you want to keep it for future reference
'''
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, test_scores = learning_curve(
    model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, test_scores_mean, label='Validation error')
plt.xlabel('Training set size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves')
plt.legend()
plt.show()
'''

# End of script