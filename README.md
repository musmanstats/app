# Titanic Data Analysis App
This is a Streamlit web application that provides interactive visualizations and predictions based on the famous Titanic dataset. The app allows users to explore various features of the dataset and predict the fare of passengers using a Random Forest model.

## Features
### Data Exploration:

View the first few rows of the Titanic dataset.
* Visualize missing values in the dataset.
* Clean missing values from the age column.
* Interactive Visualizations:

* Scatter plot with hue (selectable from the sidebar).
* Bar chart with hue (selectable from the sidebar).

### Model Training
* Select features to predict passenger fare using a Random Forest model.
* Tune hyperparameters like max_depth and n_estimators directly in the app.
* Evaluate the model with metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
