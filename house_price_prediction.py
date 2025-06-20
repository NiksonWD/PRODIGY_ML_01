import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'square_feet': [1400, 1600, 1700, 1875, 1100, 1550, 1800, 2000, 2100, 2300],
    'bedrooms':     [3, 3, 4, 4, 2, 3, 4, 4, 5, 5],
    'bathrooms':    [2, 2, 2, 3, 1, 2, 3, 3, 3, 4],
    'price':        [300000, 320000, 340000, 390000, 240000, 310000, 370000, 410000, 450000, 490000]
}

df = pd.DataFrame(data)

X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

usd_to_inr = 83

print("\nModel Coefficients (in USD):")
print("Square Feet Coefficient:", model.coef_[0])
print("Bedrooms Coefficient:", model.coef_[1])
print("Bathrooms Coefficient:", model.coef_[2])
print("Intercept:", model.intercept_)

new_house = [[1750, 3, 2]]
predicted_price_usd = model.predict(new_house)[0]
predicted_price_inr = predicted_price_usd * usd_to_inr

print(f"\nPredicted price for 1750 sq ft, 3 bed, 2 bath:")
print(f"USD: ${predicted_price_usd:,.2f}")
print(f"INR: ₹{predicted_price_inr:,.2f}")
