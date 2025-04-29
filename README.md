# Elevate-labs---task3

This repository contains the implementation of linear regression for the House Price Prediction dataset 

## Files
Housing : Dataset used for training and testing.
linear_regregression: Python script implementing linear regression.
regression_plot: Plot of actual vs. predicted prices.

## Steps
1. Loaded and preprocessed the dataset, encoding categorical variables.
2. Split data into 80% training and 20% testing sets.
3. Trained a Linear Regression model using scikit-learn.
4. Evaluated the model using mean absolute error, mean square eroor, and R^2 metrics.
5. Visualized actual vs. predicted prices for the area feature.

## Results
Mean Absolute Error (MAE): 979679.69
Mean Squared Error (MSE): 1771751116594.03
R^2 Score: 0.65

Model Coefficients:
                   Coefficient
area              2.358488e+02
bedrooms          7.857449e+04
bathrooms         1.097117e+06
stories           4.062232e+05
mainroad          3.668242e+05
guestroom         2.331468e+05
basement          3.931598e+05
hotwaterheating   6.878813e+05
airconditioning   7.855506e+05
parking           2.257565e+05
prefarea          6.299017e+05
furnishingstatus -2.103971e+05

## Tools Used
- Python, scikit-learn, pandas, matplotlib
