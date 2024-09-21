# Real Estate Data Collection and Enrichment

This project automates the collection and enrichment of real estate data, with a focus on properties in Dublin, Ireland. The scripts use web scraping to gather property information and enrich it with additional metadata from external sources like MyHome.ie and Google Maps.

## Directory Structure

The project is organized as follows:

```bash
data/
├── raw/         # Raw data scraped from real estate websites
├── processed/   # Data enriched and processed for analysis
logs/            # Logs of scraping activities and errors
notebooks/       # Jupyter notebooks for data exploration and analysis
src/
├── scrapers/    # Scripts for web scraping
├── utils/       # Utility functions for file and data processing
tests/           # Test scripts for scrapers and utilities
Main Scripts
1. find_my_home.py
Purpose: This script enriches property metadata by searching for listings on MyHome.ie. It extracts detailed property information such as address, asking price, floor area, and BER rating.
Key Features:
Performs a Google search to find relevant MyHome.ie URLs.
Scrapes MyHome.ie for property data like address, asking price, number of bedrooms, bathrooms, and BER rating.
Optionally enriches the data with latitude and longitude using the Google Maps API.
2. my_home_scrape.py
Purpose: Scrapes property data from the Dublin Price Register on Mynest.ie, gathering URLs and addresses of properties sold in Dublin.
Key Features:
Scrapes property data from multiple pages.
Captures unique addresses and associated URLs.
Stores the data in a CSV file for further enrichment and analysis.
Other Noteworthy Scripts
my_home_metadata_enrichment.py: Enriches property data with metadata such as floor area, BER details, and pricing by scraping MyHome.ie.

pagination_test_script.py: A utility script for testing pagination functionality on websites, ensuring correct handling of multi-page scrapes.

image_comparison_metrics.py: Processes and compares real estate images using various metrics, helpful for tracking changes in listings or identifying similar properties.

Dependencies
Python Libraries:
selenium: For browser automation and web scraping.
webdriver_manager: Automatically manages browser drivers.
requests: For making HTTP requests (e.g., to Google Maps API).
csv: For reading and writing CSV files.
External Services:
Google Maps API: Used to fetch latitude and longitude data for property addresses.
MyHome.ie: Used to enrich property metadata with detailed property information.
Usage
Enrich Metadata for Scraped Properties:

bash
Copy code
python src/scrapers/find_my_home.py
This will search for properties on MyHome.ie and enrich your existing dataset with additional details such as address, price, and BER rating.

Scrape Dublin Price Register:

bash
Copy code
python src/scrapers/my_home_scrape.py
This will scrape the Dublin Price Register, gathering property sale data and saving it to a CSV file.

Analyze Data in Jupyter Notebooks: Open any of the Jupyter notebooks in the notebooks/ directory to explore and visualize the scraped data.

Logging and Debugging
All scraping logs are stored in the logs/ directory. Logs include detailed messages and any errors encountered during the scraping process, making debugging easier.

Future Improvements
Add scraping capabilities for additional regions or other real estate websites.
Automate the entire pipeline using a task scheduler (e.g., cron) for continuous data collection and enrichment.