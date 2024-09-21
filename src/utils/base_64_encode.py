import os
import base64
from PIL import Image

# Directory with images
image_directory = './collages'
output_directory = './collages_base_64'

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

def encode_image_to_base64(image_path):
    """Encode an image to Base64."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def process_images(directory):
    # List of image files
    image_files = [f for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg'))]

    # Ensure there are images to process
    if len(image_files) < 1:
        print("No images found to process.")
        return

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        encoded_image = encode_image_to_base64(image_path)

        # Save the encoded image to a new file
        output_file_path = os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_base64.txt")
        with open(output_file_path, "w") as output_file:
            output_file.write(encoded_image)

        print(f"Encoded {image_file} to {output_file_path}")

# Run the image processing
process_images(image_directory)
