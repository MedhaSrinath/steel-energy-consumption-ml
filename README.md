# Steel Energy Consumption ML

A lightweight machine learning project scaffold for predicting steel industry energy consumption.

## Structure

- `data/steel_industry_data.csv` - source dataset
- `notebooks/eda.ipynb` - exploratory data analysis notebook
- `src/` - preprocessing, feature engineering, training, and prediction helpers
- `app/app.py` - Streamlit app
- `models/model.pkl` - saved trained model artifact

## Setup

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app/app.py
```

## Notes

The current code is schema-agnostic so it can be adapted to the exact dataset column names once the CSV is available.
