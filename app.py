# SECTION 7: Streamlit App Script
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(layout="wide")
st.title("🌍 Climate Change Forecast - Tanzania")
st.markdown("This Interactive dashboard ")

st.sidebar.header(" Controls : Years Navigation ")
year_range=st.sidebar.slider("Select Year Range : ", 1980,2024,(1990,2024))
forecast_years = st.sidebar.slider("Forecast Future Temp Until ",2025,2035,2030)


np.random.seed(42)
years = np.range(1980,2025)
temperature = 22 + 0.03 * (years - 1980) + np.random.normal(0,0.5,len(years))
precipitation = 800 + 2 * (years - 1980) + np.random.normal(0,30,len(years))

df = pd.DataFrame({
    'Year':years,
    'Average_Temperature_C':temperature,
    'Annual_Precipitation':precipitation
})

df_filtered = df[( df['Year']>= year_range[0]) & (df['Year']<=year_range[1])]

st.subheader("1. Sample Climate Data")
st.dataframe(df_filtered.head())

df_filtered['Year'] = pd.to_dataframe(df_filtered['Year'],format='%Y')
df_filtered.set_index('Year',inplace=True)


st.subheader("2. Descriptive Statistics")
st.write(df_filtered.describe())

