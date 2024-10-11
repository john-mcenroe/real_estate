import joblib
import pandas as pd
import os

# Load the trained model
model_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/xgboost_model.joblib'
model_pipeline = joblib.load(model_path)

def predict_sold_price(input_data):
    """
    Predict the sold price for a given input.
    
    :param input_data: A dictionary containing the feature values
    :return: The predicted sold price
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    predicted_price = model_pipeline.predict(input_df)[0]
    
    return predicted_price

# Get all required features
required_features = model_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Example usage
if __name__ == "__main__":
    # Create a sample input with all required features
    sample_input = {feature: 0 for feature in required_features}
    
    # Update some key features (you can adjust these as needed)
    sample_input.update({
        'MyHome_Beds': 3,
        'MyHome_Baths': 2,
        'MyHome_Floor_Area_Value': 120,
        'MyHome_Latitude': 53.3498,
        'MyHome_Longitude': -6.2603,
        'MyHome_Asking_Price': 350000,
        'Property Type': 'Semi-D',
        'Energy Rating': 'C2',
    })
    
    predicted_price = predict_sold_price(sample_input)
    print(f"Predicted Sold Price: €{predicted_price:,.2f}")

    # Interactive input for key features
    print("\nEnter property details:")
    key_features = ['MyHome_Beds', 'MyHome_Baths', 'MyHome_Floor_Area_Value', 'MyHome_Latitude', 'MyHome_Longitude', 'MyHome_Asking_Price', 'Property Type', 'Energy Rating']
    interactive_input = sample_input.copy()
    for key in key_features:
        value = input(f"{key}: ")
        interactive_input[key] = float(value) if key not in ['Property Type', 'Energy Rating'] else value
    
    predicted_price = predict_sold_price(interactive_input)
    print(f"\nPredicted Sold Price: €{predicted_price:,.2f}")
