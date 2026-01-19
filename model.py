#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data.csv")

print("Dataset Preview:")
print(data.head())
#this confirms data is loaded correctly

#separate features and target
# Input features
X = data[['StudyHours', 'Attendance', 'PreviousScore']]

# Target variable
y = data['FinalScore']
#ml models learn from x to predict y.

#split data(training & testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
# train data-->learn patterns
#test data---> check accuracy

#train the ml model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully")
# we use linear regression because
#-->continuous o/p (marks)
#-->simple and interview-friendly

#make predictions
y_pred = model.predict(X_test)

print("Predicted Scores:", y_pred)
print("Actual Scores:", y_test.values)

#evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
# mae--> avg err
# mse--> squared err
# R^2--> how well model fits data

#visualize results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Student Scores")
plt.show()
#straight line=good model
