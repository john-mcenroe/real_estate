import json

def extract_python_from_ipynb(ipynb_file, output_file):
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    python_code = ""
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            python_code += ''.join(cell['source']) + '\n\n'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(python_code)

    print(f"Python code extracted and saved to {output_file}")

# Example usage:
extract_python_from_ipynb('/Users/johnmcenroe/Documents/programming_misc/real_estate/notebooks/exploring_property_data_iteration_11 (xGBoost V3).ipynb', '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/extracted_python_xgboost.py')
