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

# Example usage
if __name__ == "__main__":
    # Example input data (adjust features based on your model)
    sample_input = {
        'Beds': 3,
        'Baths': 2,
        'Floor Area Value': 120,
        'Latitude': 53.3498,
        'Longitude': -6.2603,
        # Add other features as needed
    }
    
    predicted_price = predict_sold_price(sample_input)
    print(f"Predicted Sold Price: €{predicted_price:,.2f}")

    # Interactive input
    print("\nEnter property details:")
    interactive_input = {}
    for key in sample_input.keys():
        value = input(f"{key}: ")
        interactive_input[key] = float(value) if key != 'Beds' and key != 'Baths' else int(value)
    
    predicted_price = predict_sold_price(interactive_input)
    print(f"\nPredicted Sold Price: €{predicted_price:,.2f}")