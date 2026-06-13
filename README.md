# German Credit Risk Prediction using Machine Learning

## Project Overview
This project focuses on predicting credit risk using the German Credit dataset. Multiple machine learning classification models were implemented and compared to identify the best-performing model for credit risk prediction.

## Objective
- Build a machine learning model to classify customers based on credit risk.
- Analyze important factors influencing credit risk.
- Handle imbalanced data and improve model performance through optimization techniques.

## Dataset
The project uses the German Credit dataset containing customer financial and demographic attributes.  
The target variable represents credit risk classification.

## Tools & Technologies
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- XGBoost
- SMOTE

## Project Workflow

### Data Understanding & Preprocessing
- Explored dataset structure and features.
- Performed data preprocessing and feature analysis.
- Prepared data for machine learning modeling.

### Exploratory Data Analysis (EDA)
- Analyzed feature distributions and relationships.
- Visualized patterns related to credit risk.

### Machine Learning Models Implemented
The following classification models were trained and evaluated:

- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- AdaBoost
- Gradient Boosting
- XGBoost

## Handling Imbalanced Data
- Identified class imbalance in the target variable.
- Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.
- Re-trained models and compared performance.

## Model Optimization
- Performed hyperparameter tuning to improve model performance.
- Compared model results before and after tuning.

## Model Evaluation
Models were evaluated using:
- Accuracy score
- Precision
- Recall
- F1-score
- Train and test performance comparison

## Deployment
A Streamlit web application was created to allow users to enter input details and receive credit risk predictions using the trained model.

## Conclusion
This project demonstrates an end-to-end machine learning workflow including data preprocessing, exploratory analysis, handling imbalanced data, model comparison, hyperparameter tuning, and deployment.
