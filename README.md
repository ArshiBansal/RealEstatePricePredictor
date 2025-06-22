<h1 align="center">🏡 Real Estate House Price & Analytics Predictor</h1>

# Overview 📊

This Streamlit-based web application provides a comprehensive platform for real estate price predictions, market analysis, and geospatial visualizations. Leveraging advanced machine learning algorithms and historical real estate data from **2001 to 2022**, the app delivers statistical insights, property valuations, and interactive analytics for real estate professionals, investors, and developers.

# Features ⚡

- 📈 **Market Overview**: Price distributions, town comparisons, key metrics like median price and yearly growth  
- 💸 **Price Estimations**: Predict property values with ML models (Random Forest, XGBoost, etc.)  
- 📊 **Market Trends**: Analyze historical price trends, seasonal patterns, and sales volume  
- 🌍 **Geographic Analysis**: Interactive heatmaps and town-level pricing using Folium  
- 🧠 **Model Performance**: Compare R², RMSE, and MAPE across multiple models  
- 🧪 **Advanced Insights**: Correlation matrices, SHAP analysis, and raw data exploration  
- 🎛️ **Custom Filters**: Filter by town, year, price, property type, season, and more  
- 🎨 **Theme Support**: Light, Dark, and Modern Blue themes available  

# Technologies Used 🛠️

### 🐍 Python Libraries:
- Streamlit – UI interface  
- Pandas, NumPy – Data handling  
- Plotly, Seaborn – Visualizations  
- Folium – Geospatial mapping  
- Scikit-learn, XGBoost – Machine learning  
- SHAP – Model explainability  

### 📦 Machine Learning Models:
- Random Forest  
- Gradient Boosting  
- Extra Trees  
- Ridge Regression  
- XGBoost  

### 🔄 Data Processing:
- `@st.cache_data`, `@st.cache_resource` – For caching  
- `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder` – For preprocessing  

### 🧾 Logging:
- Python’s `logging` module for tracking and debugging  

# Installation 🧩

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
streamlit run app.py
```
# Data Requirements 📂

- Place `real_estate_sales.csv` in the root project directory  
- **Key Columns**:  
  - `Assessed Value`  
  - `Sale Amount`  
  - `Property Type`  
  - `Date Recorded`  
  - `Town`  
  - `Location` (optional for coordinates)  
  - `Address` (optional for property ID)  

### 🧠 Feature Engineering Includes:
- `Sales Ratio` – Ratio of Sale Amount to Assessed Value  
- `PriceToYearlyMedian` – Comparison to annual town median  
- `TownPricePremium` – Premium over town average  
- `PotentialFlip` – Flip potential flag for investments  

# Usage Guide 🚀

- **Launch App**:  
  ```bash
  streamlit run app.py
  ```
 ### 🔎 Apply Filters
Use the sidebar to filter the dataset by:
- 🏘️ **Town**
- 📅 **Year**
- 🏠 **Property Type**
- 💲 **Price Range**
- ❄️ **Season**

### 🧭 Explore Tabs
- 📈 **Market Overview** – Visual insights into price distributions and market medians  
- 💸 **Price Estimations** – Predict property prices using trained models  
- 📊 **Market Trends** – Historical price analysis and seasonal patterns  
- 🌍 **Geographic Analysis** – Heatmaps and town-level geospatial views  
- 🧠 **Model Performance** – Compare ML models and residuals  
- 🔬 **Advanced Insights** – Raw data, downloads, correlation matrix  

### 🧪 Train Models
Click the **"Train Models"** button to retrain ML models based on your current filter selections.

### 📥 Download
Export filtered datasets directly from the sidebar as `.csv` files.

---

# Notes 📝

- 📦 **Caching**  
  Cached dataset stored at: `./cache/processed_real_estate_data.pkl` for faster load times  

- 🧭 **Coordinates**  
  If `Location` is missing, fallback lat/lon values are auto-generated (e.g., within Connecticut)

- 🔢 **Training Requirements**  
  At least **50 valid records** required to enable training

- 🎨 **Themes**  
  Users can select **Light**, **Dark**, or **Modern Blue** UI themes via sidebar

- 🧼 **Error Handling**  
  - Configured with Python’s `logging` module for diagnostics  
  - Graceful fallback alerts for missing columns or corrupt records  

---

# Limitations ⚠️

- Requires a valid `real_estate_sales.csv` with all essential columns  
- Geospatial visualizations assume **U.S.-based** coordinates (default: Connecticut)  
- **SHAP interpretability** limited to the **Random Forest model** on a test subset  
- Forecasting uses a **basic seasonal model** — not optimized for long-term predictions  

---

# Future Improvements 🔮

- 🔄 Support for **real-time API feeds** or external datasets  
- ⏱️ Integration of advanced time-series forecasting (e.g., **ARIMA**, **Prophet**)  
- 📍 Add **property-level markers** and tooltips on the map  
- ⚙️ Use **parallel processing** to scale model training  
- 🔐 Add **user authentication** for saving preferences and sessions  
