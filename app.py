%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np # for numeric Processing
import matplotlib.pyplot as plt # visualization
import seaborn as sns # Visualization
from sklearn.model_selection import train_test_split # Machine Learning model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose # plot time series data

st.set_page_config(page_title="Tanzania Climate Analysis", layout="wide")
st.title(" Climate Change Analysis - Tanzania")
st.markdown("This Interactive ")

st.sidebar.header("Years Navigation")
year_range= st.sidebar.slider('Select Year', 1980, 2024,(1990,2024))
forcast_years = st.sidebar.slider('Forecast Future Temp', 2026, 2031,(2026,2031))

df = pd.read_csv('https://raw.githubusercontent.com/abuu94/AnalyticsCorner/refs/heads/main/omdenaCorner/cap-project/tanzania_climate_data.csv')
# print(df.head())
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
# df_filtered.head()
st.subheader("1. Sample Climate Data")
st.dataframe(df_filtered.head())
df_filtered['Year'] = pd.to_datetime(df_filtered['Year'], format='%Y')
df_filtered.set_index('Year', inplace=True)
st.subheader("2. Descriptive Statistics")
st.write(df_filtered.describe())
st.subheader("3. Climate Trends Over Time")
fig1,ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df_filtered.index.year, df_filtered['Average_Temperature_C'], label='Avg Temp (°C)', color='green')
ax1.plot(df_filtered.index.year, df_filtered['Total_Rainfall_mm'], label='Rainfall (mm)', color='blue')
ax1.set_title('Temperature : Climate Trends in Tanzania (1980–2025)')
ax1.set_xlabel('Year')
ax1.set_ylabel('Value')
ax1.legend()
ax1.grid()
st.pyplot(fig1)
st.subheader("4. Correlation Analysis")
fig2,ax2 = plt.subplots(figsize=(6, 4))
sns.heatmap(df_filtered.corr(), annot=True, cmap='coolwarm',ax=ax2)
ax2.set_title("Correlation Matrix")
st.pyplot(fig2)
st.subheader("5. Seasonal Decomposition")
if len(df_filtered) > 5:
  result = seasonal_decompose(df_filtered['Average_Temperature_C'], model='additive', period=5 )
  fig3 = result.plot()
  fig3.set_size_inches(10,3)
  st.pyplot(fig3)
else:
  st.warning("Not enough data points")

st.subheader("6. Temperature Forecast")
df_ml = df_filtered.copy()
df_ml['Year'] = df_ml.index.year
X = df_ml[['Year']]
y_temp = df_ml['Average_Temperature_C']

X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
lr=LinearRegression()
rf=RandomForestRegressor(n_estimators=100, random_state=42)
lr.fit(X_train, y_temp_train)
rf.fit(X_train, y_temp_train)

y_prec_lr=lr.predict(X_test)
y_prec_rf=rf.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    st.write(f"{model_name} Evaluation:")
    st.write(f" - RMSE: {rmse:.2f}")
    st.write(f" - MAE: {mae:.2f}")
    st.write()
    print(f"{model_name} Evaluation:")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - MAE: {mae:.2f}")
# future_years = pd.DataFrame({'Year': np.arange(forcast_years[0], forcast_years[1])})

evaluate_model(y_temp_test, y_prec_lr, "Linear Regression")
evaluate_model(y_temp_test, y_prec_rf, "Random Forest Regressor")

st.subheader(f"7. Forecast Future Temperature (2025-{forcast_years}) ")
future_years = pd.DataFrame({'Year': np.arange(forcast_years[0], forcast_years[1])})
future_preds = rf.predict(future_years)
st.write(future_temp_pred)

fig4,ax4 = plt.subplots(figsize=(10, 5))
ax4.plot(df_filtered.index.year, df_filtered['Average_Temperature_C'], label='Historical Temp', color='green')
ax4.plot(future_years['Year'], future_preds, label='Predicted Temp (2025-2031)', color='orange')
ax4.set_title("Temperature Forecast for Tanzania")
ax4.set_xlabel("Year")
ax4.set_ylabel("Temperature (°C)")
ax4.legend()
ax4.grid()
st.pyplot(fig4)

st.success("Done")
