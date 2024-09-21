import os
import pandas as pd
import json
import requests

# Set OpenAI API key (ensure your environment variable is correctly set)
api_key = os.getenv('OPENAI_API_KEY')

# Directory with base64 text files
image_directory = './collages_base_64'
output_csv = 'property_comparison.csv'  # Name of the output CSV file

def get_base64_strings_from_directory(directory):
    # List of base64 text files
    base64_files = [f for f in os.listdir(directory) if f.lower().endswith('.txt')]
    base64_strings = []
    property_names = []
    for base64_file in base64_files:
        property_name = os.path.splitext(base64_file)[0]
        property_names.append(property_name)
        with open(os.path.join(directory, base64_file), 'r') as f:
            base64_string = f.read().strip()
            base64_strings.append(base64_string)
    return property_names, base64_strings

def analyze_property_images(property_names, base64_strings):
    # Prepare the prompt
    prompt_text = f"""
Analyze the following properties based on their interior quality across the following traits, scoring each on a scale of 1-10. Compare them in a table where columns are the property names and rows are the traits. The properties are: {', '.join(property_names)}. The traits are:

- Overall design quality
- Furnishings and finishes
- Layout and space optimization
- Natural light and views
- Cleanliness and maintenance

Return the result in JSON format with the structure:
{{
    "PropertyName1": {{
        "Overall design quality": X,
        "Furnishings and finishes": X,
        "Layout and space optimization": X,
        "Natural light and views": X,
        "Cleanliness and maintenance": X
    }},
    "PropertyName2": {{
        ...
    }},
    ...
}}

Please output only the JSON and nothing else.
"""

    # Build the content list
    content_list = [
        {
            "type": "text",
            "text": prompt_text.strip()
        }
    ]

    for property_name, base64_string in zip(property_names, base64_strings):
        content_list.append({
            "type": "text",
            "text": f"Property name: {property_name}"
        })
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_string}"
            }
        })

    messages = [
        {
            "role": "system",
            "content": "You are an AI specialized in property evaluation based on visual images."
        },
        {
            "role": "user",
            "content": content_list
        }
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Print the response for debugging
    print("API Response:", response.text)

    # Check if the response is valid JSON
    try:
        json_data = response.json()  # Parse the JSON response

        # Check for errors in the response
        if 'error' in json_data:
            print(f"Error from API: {json_data['error']['message']}")
            return {}

        # Extract the content from the response
        content = json_data['choices'][0]['message']['content']

        # Attempt to parse the content as JSON
        result_json = json.loads(content)
        return result_json

    except (ValueError, json.JSONDecodeError) as e:
        print("Error parsing JSON:", e)
        print("Response content:", response.text)
        return {}

    except KeyError as e:
        print("KeyError:", e)
        print("Response content:", response.text)
        return {}

def process_images_from_directory(directory):
    # Get property names and base64 strings
    property_names, base64_strings = get_base64_strings_from_directory(directory)

    if not property_names:
        print("No images found to compare.")
        return

    # Analyze property images
    result_json = analyze_property_images(property_names, base64_strings)

    if not result_json:
        print("No valid data to process.")
        return

    # Convert the JSON result to a Pandas DataFrame
    try:
        df = pd.DataFrame.from_dict(result_json, orient='index')
    except Exception as e:
        print("Failed to create DataFrame:", e)
        return

    # Print the DataFrame as a table
    print("\nProperty Comparison Table:\n")
    print(df)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index_label='Property Name')
    print(f"\nData saved to {output_csv}")

    return df

# Run the image processing and print the comparison table
if __name__ == "__main__":
    process_images_from_directory(image_directory)
