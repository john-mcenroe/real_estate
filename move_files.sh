#!/bin/bash

# Define the source directory (current directory)
SOURCE_DIR=$(pwd)

# Function to move files safely
move_file() {
  if [ -f "$SOURCE_DIR/$1" ]; then
    mv "$SOURCE_DIR/$1" "$2"
    echo "Moved $1 to $2"
  else
    echo "$1 not found in $SOURCE_DIR"
  fi
}

# Move scripts related to data processing to 'src' or 'src/scrapers'
move_file "find_my_home.py" "$SOURCE_DIR/src"
move_file "download_housing_data.py" "$SOURCE_DIR/src"
move_file "my_home_scrape.py" "$SOURCE_DIR/src/scrapers"
move_file "myhome_scrape_e2e.py" "$SOURCE_DIR/src/scrapers"
move_file "long_lat_address.py" "$SOURCE_DIR/src/scrapers"
move_file "scraping_property_2_images.py" "$SOURCE_DIR/src/scrapers"
move_file "scraping_property_3_download.py" "$SOURCE_DIR/src/scrapers"
move_file "mynest_base_scrape.py" "$SOURCE_DIR/src/scrapers"
move_file "mynest_base_scrape_copy.py" "$SOURCE_DIR/src/scrapers"
move_file "mynest_scraping.py" "$SOURCE_DIR/src/scrapers"
move_file "mynest_scraping_export.py" "$SOURCE_DIR/src/scrapers"
move_file "my_home_metadata_enrichment.py" "$SOURCE_DIR/src"
move_file "myhome_image_extract.py" "$SOURCE_DIR/src/scrapers"

# Move scripts related to image processing or comparison to 'src/utils'
move_file "collage_compare.py" "$SOURCE_DIR/src/utils"
move_file "collage_compare_copy.py" "$SOURCE_DIR/src/utils"
move_file "collating_images.py" "$SOURCE_DIR/src/utils"
move_file "base_64_encode.py" "$SOURCE_DIR/src/utils"
move_file "image_comparison_metrics.csv" "$SOURCE_DIR/src/utils"
move_file "daft_images" "$SOURCE_DIR/src/utils"

# Move testing or other files to 'tests'
move_file "pagination_test_script.py" "$SOURCE_DIR/tests"

# Move notebooks to 'notebooks'
move_file "exploring_property_data.ipynb" "$SOURCE_DIR/notebooks"

# Move CSV files to 'data'
move_file "dublin_house_listings.csv" "$SOURCE_DIR/data/raw"
move_file "dublin_property_images.csv" "$SOURCE_DIR/data/raw"
move_file "output_property_data.csv" "$SOURCE_DIR/data/raw"
move_file "property_comparison.csv" "$SOURCE_DIR/data/raw"
move_file "property_data.csv" "$SOURCE_DIR/data/raw"
move_file "scraped_property_results_Dublin.csv" "$SOURCE_DIR/data/raw"
move_file "scraped_property_results_Dublin_final.csv" "$SOURCE_DIR/data/raw"

# Logs can go to the 'logs' folder (if applicable)
# Add additional moves as necessary

echo "File organization complete!"
