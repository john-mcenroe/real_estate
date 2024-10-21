import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Check scikit-learn version
import sklearn
from packaging import version
sklearn_version = sklearn.__version__
print(f"scikit-learn version: {sklearn_version}")

# =========================================
# 1. Load the Data
# =========================================
input_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/preprocess_output_test.csv'
df_with_metrics = pd.read_csv(input_path)
df = df_with_metrics.copy()

# =========================================
# 2. Preserve Relevant Columns
# =========================================
columns_to_preserve = {
    'Address': ['Address', 'MyHome_Address'],
    'Beds': ['Beds', 'MyHome_Beds'],
    'Baths': ['Baths', 'MyHome_Baths'],
    'Sold Price': ['Sold Price', 'MyHome_Sold_Price'],
    'Floor Area Value': ['MyHome_Floor_Area_Value'],
    'Latitude': ['MyHome_Latitude'],
    'Longitude': ['MyHome_Longitude'],
    'MyHome_URL': ['MyHome_Link']
}
preserved_df = pd.DataFrame(index=df.index)
for key, possible_cols in columns_to_preserve.items():
    for col in possible_cols:
        if col in df.columns:
            preserved_df[key] = df[col]
            break
    else:
        preserved_df[key] = 'N/A'

# =========================================
# 3. Drop Columns & Handle Missing Data
# =========================================
columns_to_drop = ['Address', 'MyHome_Address', 'MyHome_Link', 'Asking Price', 'Sold Asking Price', 'Sold Date', 'Local Property Tax']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

target = 'Sold Price'
X = df.drop(target, axis=1)
y = df[target]

leakage_columns = ['transaction_volume_num_sold_within_3km_90days', 'transaction_volume_avg_monthly_transactions_within_3km']
X = X.drop(columns=[col for col in leakage_columns if col in X.columns], axis=1)

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if 'Eircode' in categorical_cols:
    X = X.drop('Eircode', axis=1)
    categorical_cols.remove('Eircode')

# Ensure numerical and categorical columns are correctly typed
X[numerical_cols] = X[numerical_cols].apply(pd.to_numeric, errors='coerce')
X[categorical_cols] = X[categorical_cols].astype(str)

# Drop columns with >50% missing values
threshold = 0.5
X = X.drop(columns=X.columns[X.isnull().mean() > threshold], axis=1)
numerical_cols = [col for col in numerical_cols if col in X.columns]
categorical_cols = [col for col in categorical_cols if col in X.columns]

# =========================================
# 4. Preprocessing Pipelines
# =========================================
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', onehot_encoder)
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# =========================================
# 5. Model Pipelines (Adding New Models)
# =========================================
models = {
    'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
#    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
#    'Neural Network': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    'Lasso Regression': Lasso(alpha=0.1, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
#    'CatBoost': CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=0),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
}

# Preprocessing and model pipelines
model_pipelines = {name: Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)]) for name, model in models.items()}

# =========================================
# 6. Train and Predict with Each Model
# =========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
predictions = {}

for name, pipeline in model_pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    predictions[name] = y_pred
    print(f"\n{name} Model Performance:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")

# =========================================
# 7. Median Prediction
# =========================================
pred_df = pd.DataFrame(predictions)
median_pred = pred_df.median(axis=1)

# Evaluate the median predictions
rmse_median = np.sqrt(mean_squared_error(y_test, median_pred))
r2_median = r2_score(y_test, median_pred)
print("\nMedian Prediction Performance:")
print(f"RMSE: {rmse_median:.2f}")
print(f"R²: {r2_median:.2f}")

# =========================================
# 8. Feature Importance for Median Model
# =========================================
def get_feature_names(preprocessor, numerical_cols, categorical_cols):
    output_features = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            output_features.extend(columns)
        elif name == 'cat':
            onehot = transformer.named_steps['onehot']
            cat_features = onehot.get_feature_names_out(columns)
            output_features.extend(cat_features)
    return output_features

feature_names = get_feature_names(preprocessor, numerical_cols, categorical_cols)
model_importances = pd.DataFrame()

# Use Random Forest as an example for feature importance
forest_pipeline = model_pipelines['Random Forest']
importances = forest_pipeline.named_steps['regressor'].feature_importances_

# Display top features
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(f"\nTop 10 Feature Importances from Random Forest:")
print(feature_importances.head(10))

# =========================================
# 9. Visualize Feature Importances
# =========================================
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10), palette='viridis')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# =========================================
# 10. Combine Predictions and Save to CSV
# =========================================
preserved_test = preserved_df.loc[X_test.index]
final_test_df = preserved_test.copy()
final_test_df['Actual Sold Price'] = y_test.values
final_test_df['Median Predicted Sold Price'] = median_pred

# Calculate difference and percentage difference
final_test_df['Diff'] = final_test_df['Median Predicted Sold Price'] - final_test_df['Actual Sold Price']
final_test_df['% diff'] = (final_test_df['Diff'] / final_test_df['Actual Sold Price']) * 100

# Save to CSV
output_filename = 'final_test_predictions_median.csv'
output_path = os.path.join(os.path.dirname(input_path), output_filename)
final_test_df.to_csv(output_path, index=False)
print(f"\nSaved final test predictions to {output_path}")

# =========================================
# 11. Adding More Models in the Future
# =========================================
# To add more models, simply append them to the `models` dictionary and the script will automatically
# include them in the training and prediction process.
