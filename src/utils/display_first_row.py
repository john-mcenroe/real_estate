import pandas as pd

# Load your dataset
file_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/preprocess_output_test.csv'  # Replace with the path to your file
df = pd.read_csv(file_path)

# Select the first row
first_row = df.iloc[0]

# Create a new DataFrame to display columns and their corresponding values from the first row
formatted_df = pd.DataFrame({
    'Column': df.columns,
    'First Value': first_row.values
})

# Print the DataFrame with proper formatting for two columns
for index, row in formatted_df.iterrows():
    print(f'{row["Column"]: <70} {row["First Value"]}')
