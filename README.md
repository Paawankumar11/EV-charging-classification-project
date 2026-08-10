#⚡ EV Charging Classification Project

## Project Overview

This project uses supervised machine learning to predict whether an EV charging location is likely to have **fast DC charging** based on its location and charging infrastructure footprint.

The project is designed around a practical problem: power-rating information may be missing or unreliable, while information such as station count, port count, and geographic location is generally available.

The target variable is `has_fast_dc`, making this a binary classification problem.

## 🚀Live Application

Deployed application:

https://ev-charging-classification-project-2.onrender.com

## Dataset

The project analyzes **58,642 EV charging locations** worldwide.

The original dataset contains **18 raw columns**. After data cleaning, leakage removal, and feature engineering, a **15-feature model-ready dataset** was used for classification.

### Main information used by the model

- Latitude
- Longitude
- Station count
- Port count
- Ports per station
- Country-group features

The project also identified and removed features that were mathematically related to the target variable to reduce data leakage.

## Project Workflow

The overall workflow followed these steps:

1. Data cleaning and validation
2. Handling incorrect power-related values and outliers
3. Identification and removal of data leakage
4. Exploratory data analysis
5. Feature engineering
6. Model training and comparison
7. Hyperparameter tuning
8. Final model selection
9. Model deployment using Flask

## Machine Learning Models

Nine classification models were compared using the same dataset split and evaluation approach.

The top-performing ensemble models were further tuned using `RandomizedSearchCV`.

The final model selected for the project was:

**XGBoost (Tuned)**

### Reported Performance

| Metric | Result |
|---|---:|
| ROC-AUC | 0.806 |
| Accuracy | 75% |
| Precision (Fast DC) | 75% |
| Recall (Fast DC) | 59% |

The model was selected after comparing nine candidate algorithms and tuning the leading ensemble models.

## Web Application

The machine learning model is deployed as a Flask web application.

The application accepts the following inputs:

- Latitude
- Longitude
- Station count
- Port count
- Country

The backend then:

1. Extracts the values submitted through the form.
2. Calculates `ports_per_station`.
3. Converts the selected country code into the country representation used during training.
4. Reconstructs the expected 15-feature input structure.
5. Applies the saved scaler.
6. Passes the scaled features to the trained model.
7. Displays the prediction.

The application provides two possible outcomes:

- **High-Speed DC Fast Charging Available**
- **Standard AC / Low-Power Charging Only**

## 📁Repository Structure

```text
EV-charging-classification-project/
│
├── templates/
│   └── index.html
│
├── EV_FAST_CHARGING_CLASSIFICATION.ipynb
├── EV_Fast_DC_Project_Presentation.pptx
├── app.py
├── charging_station_ml.csv
├── model.pkl
├── requirements.txt
└── scaler.pkl
```

### File Description

| File | Description |
|---|---|
| `EV_FAST_CHARGING_CLASSIFICATION.ipynb` | Complete machine learning workflow including data preparation, EDA, feature engineering, model comparison and tuning |
| `app.py` | Flask application used to serve the trained model |
| `templates/index.html` | Front-end interface for entering charging-location information |
| `model.pkl` | Saved trained machine learning model |
| `scaler.pkl` | Saved feature scaler used before prediction |
| `charging_station_ml.csv` | Dataset used for the project |
| `requirements.txt` | Python dependencies required to run the application |
| `EV_Fast_DC_Project_Presentation.pptx` | Project presentation |

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask
- HTML
- CSS
- Jupyter Notebook
- Render

## Running the Project Locally

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd EV-charging-classification-project
```

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Make sure the required files are present

```text
app.py
model.pkl
scaler.pkl
requirements.txt
templates/index.html
```

### 4. Run the Flask application

```bash
python app.py
```

The application will run locally at:

```text
http://localhost:5000
```

## Key Findings

### Site Size Is an Important Signal

Station count and port count consistently help distinguish fast-charging locations from standard charging locations. Fast-charging adoption increases as charging sites become larger.

### Geographic Patterns Matter

Fast-charging locations show geographic clustering, particularly across Western Europe and North America. Latitude, longitude, and country information contribute useful predictive information.

### Data Leakage Removal Was Important

Several columns were mathematically related to the target variable. Removing these features helped avoid artificially inflated model performance and produced a more realistic evaluation.

## Future Improvements

- Improve fast-DC recall through threshold tuning or resampling.
- Add additional features such as traffic volume, population density, and road-network proximity.
- Explore ensemble stacking.
- Expand the deployed application into a lightweight API.
- Evaluate the model on newer charging infrastructure data.

## 👤 Project by

**Paawan Kumar**
