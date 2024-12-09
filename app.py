import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the CSV file using the updated caching method
@st.cache_data
def load_data():
    # Use relative path or absolute path to the CSV file
    data = pd.read_csv("AB_NYC_2019.csv")  # Ensure file is in the same directory
    data.drop(['name', 'id', 'host_name', 'last_review'], axis=1, inplace=True)
    data['reviews_per_month'].fillna(0, inplace=True)
    data['price_per_night'] = data['price'] / data['minimum_nights']
    return data

# Load data once in cache
data = load_data()

# Encode categorical variables
label_encoders = {}
for col in ['neighbourhood_group', 'room_type']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
X = data[['neighbourhood_group', 'room_type', 'minimum_nights', 
          'calculated_host_listings_count', 'availability_365']]
y = data['price_per_night']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Streamlit app
st.title("Airbnb Dynamic Price Prediction")
st.subheader("Input Listing Details")

# User input fields
neighbourhood_group = st.selectbox("Neighbourhood Group", label_encoders['neighbourhood_group'].classes_)
room_type = st.selectbox("Room Type", label_encoders['room_type'].classes_)
minimum_nights = st.number_input("Minimum Nights", min_value=1, value=1)
calculated_host_listings_count = st.number_input("Host Listings Count", min_value=1, value=1)
availability_365 = st.slider("Availability (Days per Year)", 0, 365, 100)


# Prepare input data for prediction
input_data = pd.DataFrame({
    'neighbourhood_group': [label_encoders['neighbourhood_group'].transform([neighbourhood_group])[0]],
    'room_type': [label_encoders['room_type'].transform([room_type])[0]],
    'minimum_nights': [minimum_nights],
    'calculated_host_listings_count': [calculated_host_listings_count],
    'availability_365': [availability_365]
})

# Prediction
if st.button("Predict Price per Night"):
    price_per_night = model.predict(input_data)
    total_price = price_per_night * minimum_nights
    st.success(f"Predicted Price per Night: ${price_per_night[0]:.2f}")
    st.success(f"Total Price for {minimum_nights} Night(s): ${total_price[0]:.2f}")




st.markdown(
    """
    <hr>
    <footer style="text-align: center; font-size: small; color: gray;">
        © 2023 Sentiment Analyser App | Created By- Chakshat Bali , Savi Chopra
    </footer>
    """,
    unsafe_allow_html=True
)


