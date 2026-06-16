# ShopSmart Purchase Prediction

## Live Demo

https://shopsmart-purchase-prediction-ze5xa6khbjo8gafmevcveb.streamlit.app/

## Overview

ShopSmart is an e-commerce purchase prediction system that uses Machine Learning to determine whether a visitor is likely to complete a purchase based on their browsing behavior.

The project is built using a Decision Tree Classifier and includes data preprocessing, model training, pruning, evaluation using F1 Score, and a Streamlit web application for predictions.

---

## Problem Statement

An e-commerce company wants to understand visitor behavior and predict whether a user session will result in a purchase.

The dataset contains 12,330 user sessions with both numerical and categorical features describing browsing activity, visitor information, and session characteristics.

The goal is to build a supervised machine learning model that predicts the Revenue variable.

---

## Dataset Features

Some important features include:

* Administrative
* Administrative_Duration
* Informational
* Informational_Duration
* ProductRelated
* ProductRelated_Duration
* BounceRates
* ExitRates
* PageValues
* SpecialDay
* Month
* OperatingSystems
* Browser
* Region
* TrafficType
* VisitorType
* Weekend

Target Variable:

* Revenue (True/False)

---

## Project Structure

shop_smart_project/

├── dataset/

│ └── shop_smart_ecommerce.csv

├── artifacts/

│ ├── preprocessor.pkl

│ └── model.pkl

├── src/

│ └── train.py

├── app.py

├── requirements.txt

├── .gitignore

└── README.md

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit
* Matplotlib
* Seaborn

---

## Machine Learning Pipeline

1. Data Loading
2. Data Cleaning
3. Feature Encoding
4. Train-Test Split
5. Decision Tree Classification
6. Cost Complexity Pruning
7. Model Evaluation
8. Model Saving
9. Streamlit Deployment

---

## Model Evaluation

Evaluation Metric:

* F1 Score

Reason:

The dataset is imbalanced, therefore F1 Score provides a better measure of performance than accuracy.

Target Benchmark:

* F1 Score > 0.55

---

## Installation

Clone the repository:

git clone https://github.com/your-username/shopsmart-purchase-prediction.git

Move into the project directory:

cd shopsmart-purchase-prediction

Install dependencies:

pip install -r requirements.txt

---

## Training the Model

Run:

python src/train.py

Saved files:

* artifacts/model.pkl
* artifacts/preprocessor.pkl

---

## Running the Application

Start the Streamlit app:

streamlit run app.py

---

## Future Improvements

* Random Forest Classifier
* XGBoost Classifier
* Hyperparameter Optimization
* Cloud Deployment
* Real-time User Analytics

---

## Author

Disha Agarwal

Machine Learning Minor Project
