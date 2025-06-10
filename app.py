# # SECTION 7: Streamlit App Script
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from statsmodels.tsa.seasonal import seasonal_decompose

# st.set_page_config(layout="wide")
# st.title("🌍 Climate Change Forecast - Tanzania")
# st.markdown("This Interactive dashboard ")

# st.sidebar.header(" Controls : Years Navigation ")
# year_range=st.sidebar.slider("Select Year Range : ", 1980,2024,(1990,2024))
# forecast_years = st.sidebar.slider("Forecast Future Temp Until ",2025,2035,2030)


# np.random.seed(42)
# years = np.range(1980,2025)
# temperature = 22 + 0.03 * (years - 1980) + np.random.normal(0,0.5,len(years))
# precipitation = 800 + 2 * (years - 1980) + np.random.normal(0,30,len(years))

# df = pd.DataFrame({
#     'Year':years,
#     'Average_Temperature_C':temperature,
#     'Annual_Precipitation':precipitation
# })

# df_filtered = df[( df['Year']>= year_range[0]) & (df['Year']<=year_range[1])]

# st.subheader("1. Sample Climate Data")
# st.dataframe(df_filtered.head())

# df_filtered['Year'] = pd.to_dataframe(df_filtered['Year'],format='%Y')
# df_filtered.set_index('Year',inplace=True)


# st.subheader("2. Descriptive Statistics")
# st.write(df_filtered.describe())








import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Tanzania Climate Analysis", layout="wide")
st.title(" Climate Change Analysis - Tanzania")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ErumAfzal/Climate-Project-in-Tanzania/main/chart.csv"
    df = pd.read_csv(url)

    # Display column names for debugging
    st.write("Available columns:", df.columns.tolist())

    # Rename for clarity
    df.rename(columns={
        'Average Mean Surface Air Temperature': 'Temperature',
        'Category': 'MonthName'
    }, inplace=True)

  # Convert MonthName to month number
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df['Month'] = df['MonthName'].map(month_map)
    df['Year'] = 2025  # Assign a default or dummy year

    # Ensure temperature is numeric
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    df.dropna(subset=['Temperature'], inplace=True)

    return df

df = load_data()

if not df.empty:
    # Sidebar Inputs
    st.sidebar.header("User Input")
    month = st.sidebar.slider('Select Month', 1, 12, 1)
    year = 2025  # fixed dummy year

    # Train model
    features = df[['Year', 'Month']]
    target = df['Temperature']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    prediction = model.predict([[year, month]])[0]

    # Display result
    st.subheader("📈 Forecasted Temperature")
    st.write(f"Predicted Avg Temperature for *{year}-{month:02d}: 🌡 **{prediction:.2f} °C*")

 # Historical Trend Chart
    if st.checkbox("📊 Show Temperature Trend by Month"):
        st.line_chart(df[['Month', 'Temperature']].set_index('Month').sort_index())
        # Footer
        st.markdown("---")
        st.caption("Data Source: [World Bank Climate Portal](https://github.com/ErumAfzal/Climate-Project-in-Tanzania)")
    else:
        st.warning("No data available to display.")
