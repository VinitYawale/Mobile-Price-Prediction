import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("phone.csv")
    return data

data = load_data()

# Sidebar
st.sidebar.title("Choose Parameters")
storage = st.sidebar.slider("Storage (GB)", min_value=0, max_value=512, step=1)
ram = st.sidebar.slider("RAM (GB)", min_value=1, max_value=16, step=1)
screen_size = st.sidebar.slider("Screen Size", min_value=4.0, max_value=7.0, step=0.1)
camera = st.sidebar.slider("Camera (MP)", min_value=2, max_value=108, step=2)

# Prepare the data
X = data[['Storage(GB)', 'RAM(GB)', 'Screen_size', 'Camera(MP)']]
y = data['Price ($)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
new_phone_features = [[storage, ram, screen_size, camera]]
predicted_price = model.predict(new_phone_features)

# Display predicted price
st.title("Phone Price Prediction")
st.write("Predicted Price: $", round(predicted_price[0], 2))

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write("Mean Squared Error:", mse)
