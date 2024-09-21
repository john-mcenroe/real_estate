from PIL import Image
import os

def create_collage(image_folder):
    """Creates a collage from all images in the given folder."""

    images = []
    total_width = 0
    max_height = 0

    # Load all images and find maximum height and total width
    for filename in os.listdir(image_folder):
        print(f"Checking file: {filename}")  # Debugging print statement

        if filename.endswith('.jpg'):
            try:
                img = Image.open(os.path.join(image_folder, filename))
                images.append(img)
                total_width += img.width
                max_height = max(max_height, img.height)
                print(f"  Added image: {filename}, size: {img.size}")  # Debugging print statement
            except Exception as e:
                print(f"  Error opening image {filename}: {e}")  # Debugging print statement

    if not images:
        print("No valid images found in the folder.")
        return None

    # Create a new blank image for the collage
    collage = Image.new('RGB', (total_width, max_height))

    # Paste each image onto the collage
    x_offset = 0
    for img in images:
        collage.paste(img, (x_offset, 0))
        x_offset += img.width

    return collage

if __name__ == "__main__":
    # Get the current directory
    image_folder = '.'  

    # Create the collage
    collage_image = create_collage(image_folder)

    if collage_image:
        # Save the collage
        collage_image.save('collage.jpg')
    else:
        print("Collage creation failed.")