 AI Data Assistant – House Price Prediction System
 Project Overview

The AI Data Assistant is an end-to-end machine learning project that predicts house prices using structured real-estate data.
The project demonstrates the complete data science lifecycle — from raw data ingestion and cleaning to model training, evaluation, and deployment readiness.

This repository is designed as a portfolio-ready project that reflects real-world data workflows rather than isolated academic tasks.

 Problem Statement
Accurately estimating house prices is a common business challenge in real estate.
Raw housing data is often messy, inconsistent, and not immediately suitable for machine learning.

The goal of this project is to:
-Clean and prepare real-world housing data
-Engineer meaningful features
-Train a regression model to predict house prices
-Evaluate model performance
-Produce a reusable, deployable prediction pipeline

 Dataset
The dataset contains housing information including:
-Property size and living area
-Number of bedrooms and bathrooms
-Location attributes
-Construction and renovation details
-Nearby amenities
-Target variable: House Price

The raw dataset is cleaned and transformed before modeling.

Project Pipeline (Week 1–8 Summary)
Week 1–2: Project Setup & Data Understanding
-Project structure created
-Dataset explored
-Target variable identified
-Initial assumptions documented

Week 3: Data Cleaning
-Handled missing values
-Removed inconsistencies
-Standardized column names
-Saved cleaned dataset (clean_house_prices.csv)

Week 4: Feature Engineering
-Removed non-useful columns
-Prepared numeric features for modeling
-Ensured model-ready dataset

Week 5: Modeling
-Defined input features (X) and target (y)
-Trained a RandomForestRegressor
-Stored trained model for reuse

Week 6: Model Evaluation
-Split data into training and testing sets
-Evaluated model performance using regression metrics
-Verified prediction stability

Week 7: Prediction Pipeline
-Built a prediction script
-Generated sample predictions
-Saved trained model as a .pkl file

Week 8: Finalisation
-Project validated end-to-end
-Errors resolved
-Repository prepared for portfolio use
-Model & Evaluation

Model Used: Random Forest Regressor

Reason: Handles non-linear relationships and mixed feature importance well

Evaluation Metrics: Standard regression metrics (e.g. error measures)

The model produces realistic price predictions and generalizes well to unseen data.

 Results
-Model successfully predicts house prices
-Trained model saved for future inference
-Pipeline runs without manual intervention
-Project ready for deployment or UI integration

 How to Run the Project
-Clone the repository
-git clone <your-repo-url>
cd AI-Data-Assistant


Install dependencies

pip install -r requirements.txt


Run individual pipeline stages

python -m app.pipeline.cleaner
python -m app.pipeline.modeler
python -m app.pipeline.evaluator
python -m app.pipeline.predictor

 Project Structure
AI Data Assistant/
│
├── app/
│   └── pipeline/
│       ├── cleaner.py
│       ├── feature_engineer.py
│       ├── modeler.py
│       ├── evaluator.py
│       └── predictor.py
│
├── data/
│   ├── raw_data.csv
│   └── clean_house_prices.csv
│
├── models/
│   └── house_price_model.pkl
│
├── README.md
└── requirements.txt

Author

Passionate Machokoto
## Career Focus
Aspiring Data Scientist / Machine Learning / AI Engineer with strong foundations in data analysis and end-to-end ML systems.
