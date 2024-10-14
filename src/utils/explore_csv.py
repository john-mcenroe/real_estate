import pandas as pd

def explore_csv(file_path):
    """
    Explore a CSV file by listing its columns and showing a sample row.
    
    Args:
    file_path (str): Path to the CSV file
    
    Returns:
    None
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # List columns
    print("Columns in the CSV file:")
    for i, column in enumerate(df.columns, 1):
        print(f"{i}. {column}")
    
    print("\nSample row (transposed):")
    # Get the first row and transpose it
    sample_row = df.iloc[0].to_dict()
    for column, value in sample_row.items():
        print(f"{column}: {value}")

# Example usage
if __name__ == "__main__":
    csv_file_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/full_run_predictions_xgboost_v3.csv'
    explore_csv(csv_file_path)
