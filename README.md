# Real Estate Data Analysis and Prediction

This project focuses on collecting, processing, and analyzing real estate data, as well as building predictive models for property prices.

## Directory Structure

```
REAL_ESTATE/
├── data/
│   └── processed/  # Contains processed data files
├── env/            # Virtual environment (not tracked in git)
├── logs/           # Log files
├── notebooks/      # Jupyter notebooks for analysis and experimentation
├── src/
│   ├── db_update/  # Scripts for database updates
│   ├── modelling/  # Scripts for data processing and model building
│   ├── scrapers/   # Web scraping scripts
│   └── utils/      # Utility functions
├── .gitignore
├── package-lock.json
├── package.json
└── README.md
```

## Components

### 1. Scrapers (`src/scrapers/`)

- Purpose: Collect property price data from various sources.
- Features:
  - Configurable to continuously grab and dedupe data.
  - Automation of scraping process (to be implemented).
- Usage:
  ```
  python src/scrapers/main_scraper.py
  ```

### 2. Modelling (`src/modelling/`)

- Purpose: Process data and fit XGBoost model for property price prediction.
- Features:
  - Data preprocessing and feature engineering.
  - XGBoost model training and evaluation.
  - Produces a joblib model file for use in the application.
- Usage:
  ```
  python src/modelling/train_model.py
  ```

### 3. DB Update (`src/db_update/`)

- Purpose: Push collected and processed data to Supabase.
- Features:
  - Defines table structure for the database.
  - Handles data insertion and updates.
- Usage:
  ```
  python src/db_update/update_supabase.py
  ```

### 4. Data Folder (`data/`)

- Contains output data at different stages of processing:
  - `data/raw/`: Raw scraped data (if applicable)
  - `data/processed/`: Cleaned and preprocessed data ready for analysis or modeling

### 5. Jupyter Notebooks (`notebooks/`)

- A suite of Jupyter notebooks for exploratory data analysis and rough experimentation.
- These notebooks provide interactive environments for:
  - Data visualization
  - Feature exploration
  - Model prototyping
  - Ad-hoc analysis tasks
- To use:
  ```
  jupyter notebook notebooks/
  ```

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/real-estate-project.git
   ```

2. Set up a virtual environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   - Create a `.env` file in the project root and add necessary API keys and database credentials.

5. Run the scraper to collect data:
   ```
   python src/scrapers/main_scraper.py
   ```

6. Process data and train the model:
   ```
   python src/modelling/train_model.py
   ```

7. Update the database with new data:
   ```
   python src/db_update/update_supabase.py
   ```

8. Explore the data and experiment using Jupyter notebooks:
   ```
   jupyter notebook notebooks/
   ```

## Future Improvements

- Implement automated scheduling for regular data scraping.
- Develop a web interface for easy interaction with the model and data.
- Expand the model to include more features and improve prediction accuracy.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
