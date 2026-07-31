import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 1: Load Dataset
data = pd.read_csv("House_Price.csv")

# Step 2: Display Dataset
print("First 5 Rows:")
print(data.head())

# Step 3: Select Features and Target
X = data[['square_feet', 'bedrooms', 'bathrooms']]
y = data['price']

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict Test Data
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("\nModel Evaluation")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))

# Step 8: Predict New House Price
new_house = pd.DataFrame({
    'square_feet': [2000],
    'bedrooms': [3],
    'bathrooms': [2]
})

predicted_price = model.predict(new_house)

print("\nPredicted Price for New House:")
print(f"₹ {predicted_price[0]:,.2f}")
