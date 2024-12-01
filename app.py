import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
data = pd.read_csv("C:\\Users\\lenovo\\Dynamic Pricing\\AB_NYC_2019.csv.csv")
data.drop(['name', 'id', 'host_name', 'last_review'], axis=1, inplace=True)
data['reviews_per_month'].fillna(0, inplace=True)

# Encode categorical variables
label_encoders = {}
for col in ['neighbourhood_group', 'room_type']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
X = data[['neighbourhood_group', 'room_type', 'minimum_nights', 
          'calculated_host_listings_count', 'availability_365']]
y = data['price']

# Train the model (using Linear Regression for simplicity)
model = LinearRegression()
model.fit(X, y)

# Streamlit app
st.title("Airbnb Dynamic Price Prediction")
st.subheader("Input Listing Details")

# User inputs
neighbourhood_group = st.selectbox("Location", label_encoders['neighbourhood_group'].classes_)
room_type = st.selectbox("Room Type", label_encoders['room_type'].classes_)
minimum_nights = st.number_input("Minimum Nights", min_value=1, value=1)
calculated_host_listings_count = st.number_input("Host Listings Count", min_value=1, value=1)
availability_365 = st.slider("Availability (Days per Year)", 0, 365, 100)

# Encode user input to match model requirements
input_data = pd.DataFrame({
    'neighbourhood_group': [label_encoders['neighbourhood_group'].transform([neighbourhood_group])[0]],
    'room_type': [label_encoders['room_type'].transform([room_type])[0]],
    'minimum_nights': [minimum_nights],
    'calculated_host_listings_count': [calculated_host_listings_count],
    'availability_365': [availability_365]
})

# Predict the price
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ${prediction[0]:.2f}")
