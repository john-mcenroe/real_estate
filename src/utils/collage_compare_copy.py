import os
import openai
import pandas as pd
import json
import base64
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

images = get_base64_strings_from_directory(image_directory)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What’s the difference between these images?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{images[1][0]}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{images[1][1]}"
          }          
        }
      ]
    }
  ],
  "max_tokens": 1000
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())