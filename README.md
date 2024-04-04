

# Laptop Price Prediction Project

## Overview

This project aims to predict laptop prices based on various features such as company, type, RAM, weight, etc. It involves exploring different regression algorithms to build predictive models and evaluating their performance.

## Dataset

The dataset used in this project contains information about different laptops, including features like company, type, RAM, weight, screen specifications, CPU, GPU, and operating system. It consists of both numerical and categorical variables.

## Steps

### 1. Data Exploration and Preprocessing

- Explored the dataset to understand the distribution and characteristics of features.
- Handled missing values by imputation or removal.
- Preprocessed the data by encoding categorical variables, scaling numerical features, and splitting the data into training and testing sets.

### 2. Model Training and Evaluation

- Trained multiple regression models, including Linear Regression, Ridge Regression, Lasso Regression, KNN, Decision Tree, SVM, Random Forest, Extra Trees, AdaBoost, Gradient Boost, XGBoost, Voting Regressor, and Stacking.
- Evaluated each model's performance using metrics like R2 score and Mean Absolute Error (MAE) on the test set.
- Selected the best performing model based on evaluation metrics.

### 3. Exporting the Model

- Exported the best performing model along with the preprocessed dataset using pickle.
- Saved the trained model and dataset for future use.

