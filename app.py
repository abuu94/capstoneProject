# SECTION 7: Streamlit App Script
import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

st.title("🌍 Climate Change Forecast - Tanzania (Africa Proxy) - Mr Abubakar")

# Load historical data
@st.cache_data # Use st.cache_data instead of st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv'
    data = pd.read_csv(url)
    # Removed filtering by Source
    data['Date'] = pd.to_datetime(data['Year']) # Corrected column name to 'Year'
    data.rename(columns={'Year': 'Date', 'Mean': 'Temperature'}, inplace=True) # Rename 'Year' to 'Date' and 'Mean' to 'Temperature'
    data.dropna(inplace=True)
    # Resample to monthly average to handle potential duplicate dates from different sources
    data = data.set_index('Date').resample('MS').mean().reset_index()
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    return data[['Year', 'Month', 'Temperature']] # Ensure only these columns are returned

df = load_data()

# Model Training
# features = df[['Year', 'Month']]
# target = df['Temperature']
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# model = RandomForestRegressor(n_estimators=100)
# model.fit(X_train, y_train)

# User Input
st.sitebar.header("User Input")
year = st.slider('Select Year', 2025, 2035, 2025)
month = st.slider('Select Month', 1, 12, 1)
# prediction = model.predict([[year, month]])[0]

st.subheader(f"Predicted Avg Temperature for {year}-{month:02d}: 🌡️ {prediction:.2f} °C")

# Plot historical data
if st.checkbox("Show Historical Data"):
    st.line_chart(df.groupby('Year')['Temperature'].mean())

  
