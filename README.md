# IPL Analytics & Prediction

An end-to-end Python, Streamlit, and machine learning project for exploring Indian Premier League match data and predicting second-innings match outcomes.

## Features

- Automatic data loading and preprocessing for `matches.csv` and `deliveries.csv`
- Missing value handling, duplicate removal, team-name standardization, and merged analytics datasets
- Interactive EDA dashboards for team performance, toss behavior, run trends, venues, chases, and scores
- Team analytics with win/loss metrics, season-wise performance, home/away split, and head-to-head comparison
- Player analytics for batting and bowling with season, team, and player filters
- Venue analytics for average innings scores, highest scores, and batting-first/chasing win rates
- Machine learning module with Logistic Regression and Random Forest model comparison
- Pickle-based model persistence at `models/match_predictor.pkl`
- Streamlit prediction interface with winning probability and confidence output
- Auto-generated project insights for top teams, players, bowlers, and venues

## Project Structure

```text
ipl_analytics_prediction/
├── data/
│   ├── matches.csv
│   └── deliveries.csv
├── models/
│   └── match_predictor.pkl
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── analytics.py
│   ├── visualization.py
│   ├── feature_engineering.py
│   └── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── assets/
```

## Dataset

Place the IPL historical datasets in the `data/` directory:

- `data/matches.csv`
- `data/deliveries.csv`

The app supports common Kaggle IPL schemas and standardizes older team names such as Delhi Daredevils, Kings XI Punjab, and Royal Challengers Bangalore.

If the CSV files are not present, the dashboard runs with clearly labeled demo data so reviewers can still open every page. Replace the demo mode with real data before submitting portfolio screenshots or final reports.

## Installation

```bash
cd ipl_analytics_prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Usage

Train or refresh the machine learning model:

```bash
python src/train_model.py
```

Run the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Deployment Note

GitHub's browser uploader rejects files larger than 25 MB. If `data/deliveries.csv` is too large, upload `data/deliveries.csv.gz` instead. The app automatically reads both `.csv` and `.csv.gz` files.

On Windows, you can also double-click `run_app.bat` or run:

```bat
run_app.bat
```

If `streamlit` is not recognized, use `python -m streamlit run app.py`. If Python reports that Streamlit is not installed, run `python -m pip install -r requirements.txt` first.

If Windows says `No module named streamlit`, install the packages into the same Python environment:

```bat
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

If `py` is not available, try:

```bat
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Screenshots

Add screenshots after running the app with real IPL data:

- Home dashboard: `assets/home_dashboard.png`
- Team analytics: `assets/team_analytics.png`
- Player analytics: `assets/player_analytics.png`
- Venue analytics: `assets/venue_analytics.png`
- Prediction page: `assets/match_prediction.png`

## Machine Learning Approach

The prediction module creates second-innings chase states from ball-by-ball data. Each row represents the live state of a chase and includes:

- Batting team
- Bowling team
- Venue
- Target score
- Current score
- Wickets lost
- Overs completed
- Runs required
- Balls remaining
- Current run rate
- Required run rate

The project trains Logistic Regression and Random Forest classifiers, evaluates both with accuracy, classification report, and confusion matrix, then saves the best model.

## Future Improvements

- Add player form features based on recent matches
- Include venue-specific toss advantage modeling
- Add Dream11/fantasy points forecasting
- Add model explainability with SHAP
- Deploy the dashboard on Streamlit Community Cloud
- Connect to a database or scheduled data pipeline
