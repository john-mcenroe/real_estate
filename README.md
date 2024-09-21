# Real Estate Data Collection and Enrichment

This project automates the collection and enrichment of real estate data, focusing on properties in Dublin, Ireland. It uses web scraping to gather property information and enrich it with additional metadata from external sources like MyHome.ie and Google Maps.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Main Scripts](#main-scripts)
- [Other Noteworthy Scripts](#other-noteworthy-scripts)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Logging and Debugging](#logging-and-debugging)
- [Future Improvements](#future-improvements)

## Directory Structure

```
.
├── data/
│   ├── raw/         # Raw data scraped from real estate websites
│   └── processed/   # Data enriched and processed for analysis
├── logs/            # Logs of scraping activities and errors
├── notebooks/       # Jupyter notebooks for data exploration and analysis
├── src/
│   ├── scrapers/    # Scripts for web scraping
│   └── utils/       # Utility functions for file and data processing
└── tests/           # Test scripts for scrapers and utilities
```

## Main Scripts

### 1. find_my_home.py

**Purpose**: Enriches property metadata by searching for listings on MyHome.ie.

**Key Features**:
- Performs Google searches to find relevant MyHome.ie URLs
- Scrapes MyHome.ie for detailed property data (address, asking price, floor area, BER rating, etc.)
- Optionally enriches data with latitude and longitude using the Google Maps API

### 2. my_home_scrape.py

**Purpose**: Scrapes property data from the Dublin Price Register on Mynest.ie.

**Key Features**:
- Scrapes property data from multiple pages
- Captures unique addresses and associated URLs
- Stores data in a CSV file for further enrichment and analysis

## Other Noteworthy Scripts

- **my_home_metadata_enrichment.py**: Enriches property data with metadata such as floor area, BER details, and pricing by scraping MyHome.ie.
- **pagination_test_script.py**: Utility script for testing pagination functionality on websites.
- **image_comparison_metrics.py**: Processes and compares real estate images using various metrics.

## Dependencies

### Python Libraries:
- selenium: Browser automation and web scraping
- webdriver_manager: Automatic management of browser drivers
- requests: Making HTTP requests (e.g., to Google Maps API)
- csv: Reading and writing CSV files

### External Services:
- Google Maps API: Fetching latitude and longitude data for property addresses
- MyHome.ie: Enriching property metadata with detailed information

## Usage

1. **Enrich Metadata for Scraped Properties**:
   ```bash
   python src/scrapers/find_my_home.py
   ```
   This will search for properties on MyHome.ie and enrich your existing dataset with additional details.

2. **Scrape Dublin Price Register**:
   ```bash
   python src/scrapers/my_home_scrape.py
   ```
   This will scrape the Dublin Price Register, gathering property sale data and saving it to a CSV file.

3. **Analyze Data in Jupyter Notebooks**: 
   Open any of the Jupyter notebooks in the `notebooks/` directory to explore and visualize the scraped data.

## Logging and Debugging

All scraping logs are stored in the `logs/` directory. Logs include detailed messages and any errors encountered during the scraping process, facilitating easier debugging.

## Future Improvements

- Add scraping capabilities for additional regions or other real estate websites
- Automate the entire pipeline using a task scheduler (e.g., cron) for continuous data collection and enrichment
- Implement data visualization tools for better insights into the Dublin real estate market