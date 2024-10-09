import os
import pandas as pd

def remove_duplicates_from_file(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Remove duplicates based on all columns
    deduped_data = df.drop_duplicates()

    # Create the output file name by appending 'deduped' to the original file name
    base_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(base_name)
    deduped_file_name = f"{file_name}_deduped{file_extension}"

    # Create the output path in the same directory
    output_path = os.path.join(os.path.dirname(file_path), deduped_file_name)

    # Save the deduplicated data to the new CSV file
    deduped_data.to_csv(output_path, index=False)

    print(f"Deduplicated file saved to: {output_path}")

# File path to your CSV file
file_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/combine_and_dedupe/manual_combined_scraped_property_results_Dublin.csv'

# Run the function
remove_duplicates_from_file(file_path)
