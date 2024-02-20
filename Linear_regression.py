import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define columns to use
columns_to_use = ["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]

# Load data
data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", usecols=columns_to_use)

# Separate predictors (X) and target variable (y)
X = data.drop(columns=["SalePrice"])
y = data["SalePrice"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("R-squared:", r_squared)

# display actual and predicted sale prices
predictions_df = pd.DataFrame({'Actual Sale Price': y_test, 'Predicted Sale Price': y_pred})
print(predictions_df)
