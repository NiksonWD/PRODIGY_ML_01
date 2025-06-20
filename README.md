# 🏠 House Price Prediction:

## 📌 Task 01: Predict House Prices Using Linear Regression

This project implements a **Linear Regression** model using Python to predict house prices based on three features:

- **Square Footage**
- **Number of Bedrooms**
- **Number of Bathrooms**

## 🧠 Objective

To develop a simple machine learning model that can estimate housing prices based on given input features, helping understand how changes in home characteristics influence price.

## 📁 Files Included

- `house_price_prediction.py`: Main Python script containing the model implementation.
- `data.csv`: Sample dataset (optional — can also be hardcoded).
- `README.md`: Project documentation.

## 📊 Dataset Description

Each data record consists of:
- `square_feet`: Area of the house in square feet
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `price`: Target value — price of the house in dollars
  
## 🧪 How It Works

1. Load and prepare the dataset.
2. Split the data into training and testing sets.
3. Train a Linear Regression model using Scikit-Learn.
4. Evaluate the model using Mean Squared Error and R² Score.
5. Make predictions using new input data.

### 🔧 Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
