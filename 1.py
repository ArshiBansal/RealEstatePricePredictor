import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
import xgboost as xgb
import shap
import logging
import os
from pathlib import Path
import calendar
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create cache directory
cache_dir = Path("./cache")
cache_dir.mkdir(exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="🏠 Real Estate House Price & Analytics Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 4px;
    }
    .plotly-chart {
        width: 100%;
        height: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #F3F4F6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
        border-bottom: 2px solid #2563EB;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Theme toggle
st.sidebar.markdown('<div style="text-align: center; padding: 12px 0;"><h2 style="color: #1E3A8A;">🏠 Real Estate Analytics</h2><p style="font-size: 0.9rem; color: #6B7280;">Advanced Real Estate Analysis & Predictions</p><hr style="margin: 10px 0;"></div>', unsafe_allow_html=True)
theme = st.sidebar.selectbox("Interface Theme", ["Light", "Dark", "Modern Blue"])
if theme == "Dark":
    st.markdown("""
    <style>
    body { background-color: #111827; color: #F9FAFB; }
    .stApp { background-color: #111827; }
    .card { background-color: #1F2937; }
    .metric-card { background-color: #111827; border-left: 4px solid #3B82F6; }
    .main-header { color: #60A5FA; }
    .sub-header { color: #93C5FD; }
    .stTabs [data-baseweb="tab"] { background-color: #374151; }
    .stTabs [aria-selected="true"] { background-color: #1F2937; border-bottom: 2px solid #3B82F6; }
    </style>
    """, unsafe_allow_html=True)
elif theme == "Modern Blue":
    st.markdown("""
    <style>
    body { background-color: #F0F9FF; color: #0F172A; }
    .stApp { background-color: #F0F9FF; }
    .card { background-color: #FFFFFF; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); }
    .metric-card { background-color: #DBEAFE; border-left: 4px solid #2563EB; }
    .main-header { color: #1E40AF; }
    .sub-header { color: #1D4ED8; }
    .stTabs [data-baseweb="tab"] { background-color: #EFF6FF; }
    .stTabs [aria-selected="true"] { background-color: #BFDBFE; border-bottom: 2px solid #1D4ED8; }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏠 Real Estate House Price & Analytics Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <p>This comprehensive platform leverages cutting-edge machine learning algorithms for precise real estate price predictions, 
    market analysis, and advanced geospatial visualizations. Drawing from robust historical data spanning 2001-2022, 
    it offers detailed statistical insights, property valuations, and interactive map analytics.</p>
    <p><strong>Ideal for:</strong> Real estate professionals, investors, market analysts, and property developers seeking data-driven decision support.</p>
</div>
""", unsafe_allow_html=True)

# Data loading with enhanced preprocessing
@st.cache_data(ttl=3600)
def load_data():
    try:
        start_time = time.time()
        processed_file = cache_dir / "processed_real_estate_data.pkl"
        if processed_file.exists():
            data = pd.read_pickle(processed_file)
            logger.info(f"Loaded cached data in {time.time() - start_time:.2f} seconds")
            # Validate coordinates after loading cached data
            if data[['Latitude', 'Longitude']].isna().any().any():
                logger.warning("NaN values found in cached coordinates, regenerating...")
                data = generate_fallback_coordinates(data)
            return data

        if not os.path.exists('real_estate_sales.csv'):
            st.error("Dataset 'real_estate_sales.csv' not found. Please upload the dataset.")
            logger.error("Dataset file missing")
            return None

        logger.info("Starting data load...")
        data = pd.read_csv('real_estate_sales.csv').sample(frac=0.01, random_state=42)
        logger.info(f"Loaded dataset with {len(data)} rows")

        # Clean column names
        data.columns = [col.strip() for col in data.columns]

        # Define required columns
        required_columns = ['Assessed Value', 'Sale Amount', 'Property Type', 'Date Recorded', 'Town']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            logger.error(f"Missing columns: {missing_columns}")
            return None

        # Clean monetary fields
        monetary_cols = ['Assessed Value', 'Sale Amount']
        for col in monetary_cols:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

        # Clean date fields
        data['Date Recorded'] = pd.to_datetime(data['Date Recorded'], format='%d-%m-%Y', errors='coerce')
        data['Year'] = data['Date Recorded'].dt.year
        data['Month'] = data['Date Recorded'].dt.month
        data['MonthName'] = data['Date Recorded'].dt.month_name()
        data['Quarter'] = data['Date Recorded'].dt.quarter
        data['Season'] = data['Month'].apply(lambda x: 
            'Winter' if x in [12, 1, 2] else 
            'Spring' if x in [3, 4, 5] else 
            'Summer' if x in [6, 7, 8] else 'Fall')
        data['DayOfWeek'] = data['Date Recorded'].dt.day_name()

        # Calculate Sales Ratio
        mask = (data['Assessed Value'] > 0) & (data['Sale Amount'] > 0)
        data.loc[mask, 'Sales Ratio'] = data.loc[mask, 'Assessed Value'] / data.loc[mask, 'Sale Amount']
        data['Sales Ratio'] = pd.to_numeric(data['Sales Ratio'], errors='coerce')

        # Extract coordinates
        if 'Location' in data.columns:
            def extract_coords(location):
                if pd.isna(location):
                    return None, None
                match = re.search(r'POINT \(([\d\.-]+)\s+([\d\.-]+)\)', str(location))
                if match:
                    try:
                        lat, lon = float(match.group(1)), float(match.group(2))
                        # Validate coordinate ranges (assuming US-based data, e.g., Connecticut)
                        if 40.0 <= lat <= 43.0 and -74.0 <= lon <= -71.0:
                            return lat, lon
                        else:
                            logger.warning(f"Invalid coordinate range: ({lat}, {lon})")
                            return None, None
                    except ValueError:
                        return None, None
                return None, None
            coords = data['Location'].apply(lambda x: pd.Series(extract_coords(x), index=['Latitude', 'Longitude']))
            data['Latitude'] = coords['Latitude']
            data['Longitude'] = coords['Longitude']
            # Replace any remaining None/NaN coordinates with fallback
            if data[['Latitude', 'Longitude']].isna().any().any():
                logger.info("NaN coordinates detected after extraction, applying fallback")
                data = generate_fallback_coordinates(data)
        else:
            logger.info("Location column missing, generating fallback coordinates")
            data = generate_fallback_coordinates(data)

        # Advanced features
        yearly_median = data.groupby('Year')['Sale Amount'].median().reset_index().rename(columns={'Sale Amount': 'YearlyMedianPrice'})
        data = pd.merge(data, yearly_median, on='Year', how='left')
        data['PriceToYearlyMedian'] = data['Sale Amount'] / data['YearlyMedianPrice']

        town_median = data.groupby('Town')['Sale Amount'].median().reset_index().rename(columns={'Sale Amount': 'TownMedianPrice'})
        data = pd.merge(data, town_median, on='Town', how='left')
        data['PriceToTownMedian'] = data['Sale Amount'] / data['TownMedianPrice']
        overall_median = data['Sale Amount'].median()
        data['TownPricePremium'] = data['TownMedianPrice'] / overall_median - 1

        property_median = data.groupby('Property Type')['Sale Amount'].median().reset_index().rename(columns={'Sale Amount': 'PropertyTypeMedianPrice'})
        data = pd.merge(data, property_median, on='Property Type', how='left')
        data['PriceToPropertyTypeMedian'] = data['Sale Amount'] / data['PropertyTypeMedianPrice']

        if all(col in data.columns for col in ['Address', 'Property Type', 'Town']):
            data['PropertyID'] = data['Town'] + '_' + data['Address'] + '_' + data['Property Type']
            data['RepeatSale'] = data.duplicated('PropertyID', keep=False)
            data = data.sort_values(['PropertyID', 'Date Recorded'])
            data['DaysSinceLastSale'] = data.groupby('PropertyID')['Date Recorded'].diff().dt.days
            data['PotentialFlip'] = (data['DaysSinceLastSale'] <= 730) & (data['DaysSinceLastSale'] > 0)
            data['PreviousSaleAmount'] = data.groupby('PropertyID')['Sale Amount'].shift(1)
            data['PriceChange'] = data['Sale Amount'] - data['PreviousSaleAmount']
            data['PriceChangePercent'] = (data['PriceChange'] / data['PreviousSaleAmount'] * 100)

        # Drop NaN in critical columns
        critical_cols = ['Sale Amount', 'Assessed Value', 'Property Type', 'Town', 'Year', 'Month', 'Sales Ratio']
        initial_len = len(data)
        data = data.dropna(subset=critical_cols)
        logger.info(f"Dropped {initial_len - len(data)} rows with missing critical values")

        # Filter outliers
        Q1 = data['Sale Amount'].quantile(0.01)
        Q3 = data['Sale Amount'].quantile(0.99)
        IQR = Q3 - Q1
        data = data[(data['Sale Amount'] >= max(Q1 - 1.5 * IQR, 1000)) & (data['Sale Amount'] <= Q3 + 1.5 * IQR)]

        # Final NaN check for critical columns and coordinates
        if data[critical_cols + ['Latitude', 'Longitude']].isna().any().any():
            logger.warning("NaN values persist after cleaning, applying fallback coordinates")
            data = generate_fallback_coordinates(data)

        data.to_pickle(processed_file)
        logger.info(f"Processed dataset with {len(data)} rows in {time.time() - start_time:.2f} seconds")
        return data

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading failed: {str(e)}")
        return None

def generate_fallback_coordinates(data):
    logger.info("Generating fallback coordinates")
    data = data.copy()
    num_rows = len(data)
    # Ensure no NaN values in generated coordinates
    data['Latitude'] = np.where(
        data['Latitude'].isna(),
        np.random.uniform(41.0, 42.0, num_rows),
        data['Latitude']
    )
    data['Longitude'] = np.where(
        data['Longitude'].isna(),
        np.random.uniform(-73.7, -71.8, num_rows),
        data['Longitude']
    )
    # Validate generated coordinates
    data['Latitude'] = data['Latitude'].clip(lower=41.0, upper=42.0)
    data['Longitude'] = data['Longitude'].clip(lower=-73.7, upper=-71.8)
    return data

# Model preparation
def prepare_advanced_model_data(data):
    categorical_features = ['Town', 'Property Type', 'Season']
    numeric_features = ['Assessed Value', 'PriceToYearlyMedian', 'TownPricePremium', 'PriceToPropertyTypeMedian']
    date_features = ['Year', 'Month']
    all_features = categorical_features + numeric_features + date_features
    X = data[all_features].copy()
    y = data['Sale Amount']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('date', 'passthrough', date_features)
        ])

    return X, y, preprocessor

@st.cache_resource
def train_advanced_models(data):
    try:
        start_time = time.time()
        st.write("Starting model training...")
        X, y, preprocessor = prepare_advanced_model_data(data)

        valid_idx = y.notna()
        y = y[valid_idx]
        X = X.loc[valid_idx]

        if len(X) < 50:
            logger.warning("Insufficient data for model training")
            st.error("Not enough valid data for training (minimum 50 samples required).")
            return None, None, None, None, None

        price_bins = pd.qcut(y, 5, labels=False, duplicates='drop')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=price_bins)

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42)
        }

        pipelines = {name: Pipeline([('preprocessor', preprocessor), ('model', model)]) for name, model in models.items()}
        results = {}
        feature_importances = {}
        shap_values = {}

        for name, pipeline in pipelines.items():
            step_start = time.time()
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            if name in ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'XGBoost']:
                tree_preds = np.array([tree.predict(preprocessor.transform(X_test))
                                      for tree in pipeline.named_steps['model'].estimators_]) if name != 'Gradient Boosting' else None
                lower_bound = np.percentile(tree_preds, 5, axis=0) if tree_preds is not None else y_pred - 1.96 * np.std(y_test - y_pred)
                upper_bound = np.percentile(tree_preds, 95, axis=0) if tree_preds is not None else y_pred + 1.96 * np.std(y_test - y_pred)
            else:
                residuals = y_test - y_pred
                residual_std = np.std(residuals)
                lower_bound = y_pred - 1.96 * residual_std
                upper_bound = y_pred + 1.96 * residual_std

            within_interval = ((y_test >= lower_bound) & (y_test <= upper_bound)).mean() * 100

            results[name] = {
                'pipeline': pipeline,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'y_test': y_test,
                'y_pred': y_pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'prediction_interval_coverage': within_interval,
                'training_time': time.time() - step_start
            }

            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                feature_importances[name] = pipeline.named_steps['model'].feature_importances_

            if name == 'Random Forest' and len(X_test) > 0:
                try:
                    X_shap = X_test.iloc[:min(100, len(X_test))]
                    X_shap_processed = preprocessor.transform(X_shap)
                    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
                    shap_values[name] = explainer.shap_values(X_shap_processed)
                except Exception as e:
                    logger.warning(f"SHAP calculation failed for {name}: {str(e)}")
                    shap_values[name] = None

        best_model = min(results.items(), key=lambda x: x[1]['rmse']) if results else (None, None)
        logger.info(f"Trained models in {time.time() - start_time:.2f} seconds, best model: {best_model[0]}")
        return results, best_model[0], preprocessor, feature_importances, shap_values

    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        logger.error(f"Model training failed: {str(e)}")
        return None, None, None, None, None

# Time series forecasting
def forecast_market_trends(data, periods=12):
    try:
        monthly_avg = data.groupby(['Year', 'Month'])['Sale Amount'].mean().reset_index()
        monthly_avg['Date'] = pd.to_datetime(monthly_avg['Year'].astype(str) + '-' + monthly_avg['Month'].astype(str), format='%Y-%m')
        monthly_avg = monthly_avg.set_index('Date').sort_index()
        last_date = monthly_avg.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=30), periods=periods, freq='M')
        trend = np.linspace(0, 0.2, periods) * monthly_avg['Sale Amount'].mean()
        seasonality = 0.1 * monthly_avg['Sale Amount'].mean() * np.sin(np.linspace(0, 2*np.pi, periods))
        forecast = monthly_avg['Sale Amount'].iloc[-1] + trend + seasonality
        return monthly_avg, pd.DataFrame({
            'Sale Amount': forecast,
            'Lower Bound': forecast * 0.85,
            'Upper Bound': forecast * 1.15
        }, index=forecast_dates)
    except Exception as e:
        logger.error(f"Forecasting failed: {str(e)}")
        return None, None

# Market segmentation
def segment_market(data):
    try:
        cluster_features = ['Sale Amount', 'Assessed Value', 'PriceToYearlyMedian']
        cluster_data = data[cluster_features].copy().dropna()
        if len(cluster_data) < 4:
            logger.warning("Insufficient data for clustering")
            return data, None
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        segmented_data = data.loc[cluster_data.index].copy()
        segmented_data['Cluster'] = clusters
        segment_profiles = segmented_data.groupby('Cluster').agg({
            'Sale Amount': ['mean', 'median', 'count'],
            'Assessed Value': ['mean', 'median']
        }).reset_index()
        segment_profiles.columns = ['_'.join(col).strip('_') for col in segment_profiles.columns.values]
        price_ranking = segment_profiles.sort_values('Sale Amount_mean').reset_index()
        segment_names = ['Budget', 'Standard', 'Premium', 'Luxury']
        mapping = dict(zip(price_ranking['Cluster'], segment_names))
        segmented_data['Segment'] = segmented_data['Cluster'].map(mapping)
        segment_profiles['Segment'] = segment_profiles['Cluster'].map(mapping)
        return segmented_data, segment_profiles
    except Exception as e:
        logger.error(f"Market segmentation failed: {str(e)}")
        return data, None

# Load data
with st.spinner('Loading and optimizing data for advanced analytics...'):
    data = load_data()

if data is None or data.empty:
    st.markdown('<p class="error-message">No valid data available. Please check the dataset and try again.</p>', unsafe_allow_html=True)
    st.stop()

# Sidebar filters
st.sidebar.markdown('<div class="sub-header">Search Filters</div>', unsafe_allow_html=True)

with st.sidebar.expander("Location Filters", expanded=True):
    towns = sorted(data['Town'].unique())
    selected_towns = st.multiselect("Towns", towns, default=[towns[0]] if towns else [])
    property_types = sorted(data['Property Type'].unique())
    selected_property_types = st.multiselect("Property Types", property_types, default=[property_types[0]] if property_types else [])
    if 'Residential Type' in data.columns:
        residential_types = sorted(data['Residential Type'].dropna().unique())
        selected_residential_type = st.multiselect("Residential Types", residential_types, default=[])
    else:
        selected_residential_type = []

with st.sidebar.expander("Price & Time Filters", expanded=True):
    min_price = int(data['Sale Amount'].min())
    max_price = int(data['Sale Amount'].max())
    price_scale = st.radio("Price Scale", ["Linear", "Logarithmic"], horizontal=True)
    if price_scale == "Logarithmic":
        log_min = np.log10(max(min_price, 1))
        log_max = np.log10(max_price)
        log_price_range = st.slider("Price Range (Log)", float(log_min), float(log_max), 
                                   (float(log_min), float(log_max)), step=0.1, format="%.1f")
        price_range = (int(10**log_price_range[0]), int(10**log_price_range[1]))
        st.sidebar.write(f"Price: ${price_range[0]:,} - ${price_range[1]:,}")
    else:
        price_range = st.slider("Price Range ($)", min_price, max_price, (min_price, max_price))
    years = sorted(data['Year'].dropna().unique())
    year_range = st.slider("Year Range", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))
    months = sorted(data['Month'].dropna().astype(int).unique())
    month_names = [calendar.month_name[m] for m in months if 1 <= m <= 12]
    selected_months = st.multiselect("Months", month_names, default=[])
    seasons = sorted(data['Season'].dropna().unique())
    selected_seasons = st.multiselect("Seasons", seasons, default=[])

with st.sidebar.expander("Advanced Filters", expanded=False):
    include_flips = st.checkbox("Include Property Flips", value=True)
    include_repeat_sales = st.checkbox("Include Repeat Sales", value=True)
    min_ratio = float(data['Sales Ratio'].min())
    max_ratio = float(data['Sales Ratio'].max())
    ratio_range = st.slider("Sales Ratio Range", min_ratio, max_ratio, (min_ratio, max_ratio))

if st.sidebar.button("Reset All Filters", key="reset_button", use_container_width=True):
    selected_towns = []
    selected_property_types = []
    selected_residential_type = []
    selected_months = []
    selected_seasons = []
    price_range = (min_price, max_price)
    year_range = (int(min(years)), int(max(years)))
    ratio_range = (min_ratio, max_ratio)
    include_flips = True
    include_repeat_sales = True

# Apply filters
if st.sidebar.button("Run Analysis", type="primary"):
    with st.spinner('Applying filters...'):
        filtered_data = data.copy()
        if selected_towns:
            filtered_data = filtered_data[filtered_data['Town'].isin(selected_towns)]
        if selected_property_types:
            filtered_data = filtered_data[filtered_data['Property Type'].isin(selected_property_types)]
        if selected_residential_type and 'Residential Type' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Residential Type'].isin(selected_residential_type)]
        filtered_data = filtered_data[
            (filtered_data['Sale Amount'] >= price_range[0]) &
            (filtered_data['Sale Amount'] <= price_range[1]) &
            (filtered_data['Year'] >= year_range[0]) &
            (filtered_data['Year'] <= year_range[1])
        ]
        if selected_months and 'MonthName' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['MonthName'].isin(selected_months)]
        if selected_seasons and 'Season' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Season'].isin(selected_seasons)]
        if 'Sales Ratio' in filtered_data.columns:
            filtered_data = filtered_data[
                (filtered_data['Sales Ratio'] >= ratio_range[0]) &
                (filtered_data['Sales Ratio'] <= ratio_range[1])
            ]
        if 'PotentialFlip' in filtered_data.columns and not include_flips:
            filtered_data = filtered_data[~filtered_data['PotentialFlip']]
        if 'RepeatSale' in filtered_data.columns and not include_repeat_sales:
            filtered_data = filtered_data[~filtered_data['RepeatSale']]

        if filtered_data.empty:
            st.markdown('<p class="error-message">No data matches the selected filters. Please adjust your criteria.</p>', unsafe_allow_html=True)
            logger.warning("Filtered data is empty")
            st.stop()

        # Ensure no NaN coordinates in filtered data
        if filtered_data[['Latitude', 'Longitude']].isna().any().any():
            logger.info("NaN coordinates detected in filtered data, applying fallback")
            filtered_data = generate_fallback_coordinates(filtered_data)

        st.session_state['filtered_data'] = filtered_data
        logger.info(f"Filtered dataset to {len(filtered_data)} rows")

# Use filtered data
filtered_data = st.session_state.get('filtered_data', data)

# Display metrics
st.markdown('<div class="sub-header">Dataset Insights</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Properties", f"{len(filtered_data):,}")
col2.metric("Median Price", f"${filtered_data['Sale Amount'].median():,.0f}")
col3.metric("Towns", f"{filtered_data['Town'].nunique()}")
col4.metric("Property Types", f"{filtered_data['Property Type'].nunique()}")

# Main tabs
if len(filtered_data) >= 50:
    tabs = st.tabs([
        "📊 Market Overview", "🔮 Price Estimations", "📈 Market Trends",
        "🌎 Geographic Analysis", "📉 Model Performance", "💡 Advanced Insights"
    ])

    with tabs[0]:
        st.markdown('<div class="sub-header">📊 Market Overview</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Price", f"${filtered_data['Sale Amount'].mean():,.0f}")
            st.metric("Price Volatility", f"{filtered_data['Sale Amount'].std() / filtered_data['Sale Amount'].mean() * 100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Median Price", f"${filtered_data['Sale Amount'].median():,.0f}")
            st.metric("Price Range", f"${filtered_data['Sale Amount'].quantile(0.25):,.0f} - ${filtered_data['Sale Amount'].quantile(0.75):,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            yearly_growth = filtered_data.groupby('Year')['Sale Amount'].median().pct_change() * 100
            avg_yearly_growth = yearly_growth.mean()
            st.metric("Avg. Yearly Growth", f"{avg_yearly_growth:.1f}%")
            repeat_sales_pct = filtered_data['RepeatSale'].mean() * 100
            st.metric("Repeat Sales", f"{repeat_sales_pct:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sub-header">Price Distribution</div>', unsafe_allow_html=True)
        price_dist_chart_type = st.radio("Select Distribution Chart", ["Histogram", "KDE", "Box Plot"], horizontal=True)
        if price_dist_chart_type == "Histogram":
            fig = px.histogram(
                filtered_data, x="Sale Amount", nbins=50,
                color="Property Type" if len(filtered_data['Property Type'].unique()) <= 5 else None,
                marginal="box", opacity=0.7, title="Price Distribution by Property Type"
            )
            fig.update_layout(xaxis_title="Sale Amount ($)", yaxis_title="Count", legend_title="Property Type", height=500)
            st.plotly_chart(fig, use_container_width=True)
        elif price_dist_chart_type == "KDE":
            import seaborn as sns
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            prop_types = filtered_data['Property Type'].unique()
            if len(prop_types) <= 5:
                for prop_type in prop_types:
                    subset = filtered_data[filtered_data['Property Type'] == prop_type]
                    sns.kdeplot(subset['Sale Amount'], label=prop_type, ax=ax, fill=True, alpha=0.3)
            else:
                sns.kdeplot(filtered_data['Sale Amount'], ax=ax, fill=True)
            ax.set_title("Price Density Distribution")
            ax.set_xlabel("Sale Amount ($)")
            ax.set_ylabel("Density")
            if len(prop_types) <= 5:
                ax.legend(title="Property Type")
            st.pyplot(fig)
            plt.close(fig)
        else:
            fig = px.box(
                filtered_data, x="Property Type" if len(filtered_data['Property Type'].unique()) <= 8 else "Town",
                y="Sale Amount", color="Property Type" if len(filtered_data['Property Type'].unique()) <= 8 else None,
                title="Price Distribution by Property Type", points="outliers"
            )
            fig.update_layout(xaxis_title="Property Type" if len(filtered_data['Property Type'].unique()) <= 8 else "Town",
                              yaxis_title="Sale Amount ($)", height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sub-header">Town Price Comparison</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            if len(filtered_data['Town'].unique()) <= 15:
                fig = px.bar(
                    filtered_data.groupby('Town')['Sale Amount'].median().reset_index(),
                    x='Town', y='Sale Amount', color='Town',
                    labels={'Sale Amount': 'Median Sale Price ($)'}, title="Median Sale Price by Town"
                )
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Too many towns to display. Please filter your data.")
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Top Towns by Price**")
            top_towns = filtered_data.groupby('Town')['Sale Amount'].median().sort_values(ascending=False).head(5)
            for town, price in top_towns.items():
                st.markdown(f"• {town}: ${price:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Most Active Towns**")
            active_towns = filtered_data['Town'].value_counts().head(5)
            for town, count in active_towns.items():
                st.markdown(f"• {town}: {count:,} sales")
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="sub-header">🔮 Price Estimations</div>', unsafe_allow_html=True)
        if 'model_results' in st.session_state and st.session_state.model_results:
            st.markdown(f'<div class="card"><p>Using {st.session_state.best_model_name} with R² Score: {st.session_state.model_results[st.session_state.best_model_name]["r2"]*100:.1f}%</p></div>', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Property Value Estimator")
                with st.form("property_value_form"):
                    town = st.selectbox("Town", sorted(filtered_data['Town'].unique()))
                    prop_type = st.selectbox("Property Type", sorted(filtered_data['Property Type'].unique()))
                    assessed_value = st.number_input("Assessed Value", value=int(filtered_data[filtered_data['Town'] == town]['Assessed Value'].median()), step=1000)
                    year = st.selectbox("Year", sorted(filtered_data['Year'].unique()))
                    month = st.selectbox("Month", range(1, 13))
                    season = st.selectbox("Season", sorted(filtered_data['Season'].unique()))
                    submit_button = st.form_submit_button("Estimate Value")
                if submit_button and 'model_results' in st.session_state:
                    sample = pd.DataFrame({
                        'Town': [town], 'Property Type': [prop_type], 'Assessed Value': [assessed_value],
                        'Year': [year], 'Month': [month], 'Season': [season],
                        'PriceToYearlyMedian': [filtered_data['PriceToYearlyMedian'].median()],
                        'TownPricePremium': [filtered_data['TownPricePremium'].median()],
                        'PriceToPropertyTypeMedian': [filtered_data['PriceToPropertyTypeMedian'].median()]
                    })
                    model = st.session_state.model_results[st.session_state.best_model_name]['pipeline']
                    prediction = model.predict(sample)[0]
                    prediction_interval = st.session_state.model_results[st.session_state.best_model_name]['mape'] / 100 * prediction
                    lower_bound = prediction - prediction_interval
                    upper_bound = prediction + prediction_interval
                    st.success(f"Estimated Value: ${prediction:,.0f}")
                    st.info(f"Prediction Range: ${lower_bound:,.0f} - ${upper_bound:,.0f}")
                    comparables = filtered_data[
                        (filtered_data['Town'] == town) & (filtered_data['Property Type'] == prop_type)
                    ].copy()
                    if not comparables.empty:
                        comparables['SimilarityScore'] = abs(comparables['Sale Amount'] - prediction) / prediction
                        similar_props = comparables.sort_values('SimilarityScore').head(5)
                        st.markdown("##### Comparable Properties")
                        for _, prop in similar_props.iterrows():
                            st.markdown(f"• ${prop['Sale Amount']:,.0f} - {prop['Town']}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### What Impacts Property Values?")
                if 'feature_importances' in st.session_state and st.session_state.best_model_name in st.session_state.feature_importances:
                    feature_names = st.session_state.preprocessor.get_feature_names_out()
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': st.session_state.feature_importances[st.session_state.best_model_name]
                    })
                    importance_df['Feature'] = importance_df['Feature'].str.replace('cat__', '').str.replace('num__', '')
                    fig = px.bar(importance_df.sort_values('Importance', ascending=False).head(10),
                                 x='Importance', y='Feature', orientation='h', title="Top Factors Affecting Property Value")
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                if 'shap_values' in st.session_state and st.session_state.shap_values.get(st.session_state.best_model_name):
                    st.markdown("#### Property Value Drivers (SHAP Analysis)")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(st.session_state.shap_values[st.session_state.best_model_name],
                                     feature_names=feature_names, plot_type="bar", show=False)
                    st.pyplot(fig)
                    plt.close(fig)
                st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="sub-header">📈 Market Trends</div>', unsafe_allow_html=True)
        st.markdown("#### Price Trends Over Time")
        trend_data = filtered_data.groupby('Year')['Sale Amount'].agg(['median', 'mean', 'count']).reset_index()
        trend_data = trend_data.rename(columns={'median': 'Median Price', 'mean': 'Average Price', 'count': 'Sales Volume'})
        col1, col2 = st.columns([3, 1])
        with col1:
            trend_metric = st.radio("Price Metric", ["Median Price", "Average Price"], horizontal=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_data['Year'], y=trend_data[trend_metric], mode='lines+markers', name=trend_metric, line=dict(color='#2563EB', width=3)))
            fig.add_trace(go.Bar(x=trend_data['Year'], y=trend_data['Sales Volume'], name='Sales Volume', marker_color='rgba(37, 99, 235, 0.2)', opacity=0.7, yaxis='y2'))
            fig.update_layout(
                title=f"{trend_metric} and Sales Volume by Year",
                xaxis=dict(title='Year'),
                yaxis=dict(title=dict(text=f"{trend_metric} ($)", font=dict(color='#2563EB')), tickfont=dict(color='#2563EB')),
                yaxis2=dict(title=dict(text='Sales Volume', font=dict(color='#64748B')), tickfont=dict(color='#64748B'), anchor='x', overlaying='y', side='right'),
                height=500, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Market Summary**")
            first_year = trend_data['Year'].min()
            last_year = trend_data['Year'].max()
            first_price = trend_data[trend_data['Year'] == first_year][trend_metric].values[0]
            last_price = trend_data[trend_data['Year'] == last_year][trend_metric].values[0]
            total_growth = (last_price / first_price - 1) * 100
            years_diff = last_year - first_year
            annualized_growth = ((last_price / first_price) ** (1 / max(1, years_diff)) - 1) * 100
            st.metric("Total Growth", f"{total_growth:.1f}%")
            st.metric("Annualized Growth", f"{annualized_growth:.1f}%")
            if len(trend_data) >= 3:
                recent_data = trend_data.iloc[-3:]
                recent_trend = (recent_data[trend_metric].iloc[-1] / recent_data[trend_metric].iloc[0] - 1) * 100
                st.metric("Recent Trend (3 years)", f"{recent_trend:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Highest Growth Years**")
            trend_data['Growth'] = trend_data[trend_metric].pct_change() * 100
            top_growth_years = trend_data.dropna().sort_values('Growth', ascending=False).head(3)
            for _, year_data in top_growth_years.iterrows():
                st.markdown(f"• {int(year_data['Year'])}: {year_data['Growth']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### Seasonal Trends")
        seasonal_data = filtered_data.groupby('Month')['Sale Amount'].median().reset_index()
        seasonal_data['Month'] = seasonal_data['Month'].astype(int)
        seasonal_data['MonthName'] = seasonal_data['Month'].apply(lambda x: calendar.month_name[x])
        seasonal_data = seasonal_data.sort_values('Month')
        fig = px.line(seasonal_data, x='MonthName', y='Sale Amount', markers=True,
                      title="Median Sale Price by Month", labels={'Sale Amount': 'Median Sale Price ($)', 'MonthName': 'Month'})
        seasonal_data['Change'] = seasonal_data['Sale Amount'].pct_change() * 100
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Peak Season**")
            peak_month_idx = seasonal_data['Sale Amount'].idxmax()
            peak_month = seasonal_data.loc[peak_month_idx, 'MonthName']
            peak_price = seasonal_data.loc[peak_month_idx, 'Sale Amount']
            trough_month_idx = seasonal_data['Sale Amount'].idxmin()
            trough_month = seasonal_data.loc[trough_month_idx, 'MonthName']
            trough_price = seasonal_data.loc[trough_month_idx, 'Sale Amount']
            seasonal_diff = (peak_price / trough_price - 1) * 100
            st.markdown(f"**Peak Month:** {peak_month}")
            st.markdown(f"**Lowest Month:** {trough_month}")
            st.markdown(f"**Seasonal Price Difference:** {seasonal_diff:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Sales Volume by Season**")
            season_counts = filtered_data['Season'].value_counts()
            fig = px.pie(values=season_counts.values, names=season_counts.index, title="Sales Distribution by Season")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="sub-header">🌎 Geographic Analysis</div>', unsafe_allow_html=True)
        geo_data = filtered_data[['Latitude', 'Longitude', 'Sale Amount']].dropna()
        town_prices = filtered_data.groupby('Town')['Sale Amount'].median().reset_index()
        town_counts = filtered_data['Town'].value_counts().reset_index()
        town_counts.columns = ['Town', 'Count']
        town_data = pd.merge(town_prices, town_counts, on='Town')
        town_coords = {town: (41.6 + i * 0.01, -72.7 + i * 0.01) for i, town in enumerate(town_data['Town'])}  # Placeholder
        if len(geo_data) >= 10 or len(town_data) > 0:
            m = folium.Map(location=[41.6, -72.7], zoom_start=9, tiles="OpenStreetMap")
            if len(geo_data) >= 10:
                # Additional validation to ensure no NaN values in heat_data
                geo_data = geo_data[geo_data[['Latitude', 'Longitude']].notna().all(axis=1)]
                if len(geo_data) >= 10:
                    heat_data = [[row['Latitude'], row['Longitude'], row['Sale Amount']] for _, row in geo_data.sample(min(100, len(geo_data))).iterrows()]
                    HeatMap(heat_data, radius=15).add_to(m)
                else:
                    st.warning("Insufficient valid geospatial data for heatmap after removing NaN values.")
            min_price = town_data['Sale Amount'].min()
            max_price = town_data['Sale Amount'].max()
            price_range = max_price - min_price
            for _, row in town_data.iterrows():
                town = row['Town']
                price = row['Sale Amount']
                norm_price = (price - min_price) / price_range if price_range > 0 else 0.5
                color = 'red' if norm_price >= 0.75 else 'orange' if norm_price >= 0.5 else 'yellow' if norm_price >= 0.25 else 'green'
                lat, lon = town_coords.get(town, (41.6, -72.7))
                folium.CircleMarker(
                    location=[lat, lon], radius=10, color=color, fill=True, fill_opacity=0.7,
                    tooltip=f"{town}: ${price:,.0f} ({row['Count']} sales)"
                ).add_to(m)
            st.markdown("#### Town Price Map")
            folium_static(m, width=1000, height=600)
            st.markdown("#### Town Price Statistics")
            town_data_display = town_data.copy()
            town_data_display['Sale Amount'] = town_data_display['Sale Amount'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(town_data_display, hide_index=True)
        else:
            st.warning("Insufficient geospatial or town data.")

    with tabs[4]:
        st.markdown('<div class="sub-header">📉 Model Performance</div>', unsafe_allow_html=True)
        if 'model_results' in st.session_state and st.session_state.model_results:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Model", st.session_state.best_model_name)
            with col2:
                st.metric("R² Score", f"{st.session_state.model_results[st.session_state.best_model_name]['r2'] * 100:.2f}%")
            with col3:
                st.metric("RMSE", f"{st.session_state.model_results[st.session_state.best_model_name]['rmse']:,.0f}")
            with col4:
                st.metric("MAPE", f"{st.session_state.model_results[st.session_state.best_model_name]['mape']:.2f}%")
            st.markdown("#### Model Comparison")
            model_comparison = pd.DataFrame({
                'Model': list(st.session_state.model_results.keys()),
                'R² Score': [result['r2'] * 100 for result in st.session_state.model_results.values()],
                'RMSE': [result['rmse'] for result in st.session_state.model_results.values()],
                'MAE': [result['mae'] for result in st.session_state.model_results.values()],
                'MAPE (%)': [result['mape'] for result in st.session_state.model_results.values()],
                'Interval Coverage (%)': [result['prediction_interval_coverage'] for result in st.session_state.model_results.values()],
                'Training Time (s)': [result['training_time'] for result in st.session_state.model_results.values()]
            })
            model_comparison['R² Score'] = model_comparison['R² Score'].apply(lambda x: f"{x:.2f}%")
            model_comparison['RMSE'] = model_comparison['RMSE'].apply(lambda x: f"{x:,.0f}")
            model_comparison['MAE'] = model_comparison['MAE'].apply(lambda x: f"{x:,.0f}")
            model_comparison['MAPE (%)'] = model_comparison['MAPE (%)'].apply(lambda x: f"{x:.2f}%")
            model_comparison['Interval Coverage (%)'] = model_comparison['Interval Coverage (%)'].apply(lambda x: f"{x:.2f}%")
            model_comparison['Training Time (s)'] = model_comparison['Training Time (s)'].apply(lambda x: f"{x:.2f}")
            st.dataframe(model_comparison, hide_index=True)
            st.markdown("#### Prediction Accuracy")
            indices = np.random.choice(len(st.session_state.model_results[st.session_state.best_model_name]['y_test']), 1000, replace=False) if len(st.session_state.model_results[st.session_state.best_model_name]['y_test']) > 1000 else range(len(st.session_state.model_results[st.session_state.best_model_name]['y_test']))
            y_test_sample = st.session_state.model_results[st.session_state.best_model_name]['y_test'].iloc[indices]
            y_pred_sample = st.session_state.model_results[st.session_state.best_model_name]['y_pred'][indices]
            lower_bound_sample = st.session_state.model_results[st.session_state.best_model_name]['lower_bound'][indices]
            upper_bound_sample = st.session_state.model_results[st.session_state.best_model_name]['upper_bound'][indices]
            fig = go.Figure()
            max_val = max(y_test_sample.max(), y_pred_sample.max())
            min_val = min(y_test_sample.min(), y_pred_sample.min())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Prediction', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=y_test_sample, y=y_pred_sample, mode='markers', name='Predictions', marker=dict(color='#2563EB', size=8, opacity=0.6)))
            fig.update_layout(title="Predicted vs Actual Sale Prices", xaxis=dict(title="Actual Price ($)"),
                              yaxis=dict(title=dict(text="Predicted Price ($)", font=dict(color='#2563EB'))),
                              height=600, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Residual Analysis")
            residuals = y_test_sample - y_pred_sample
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(residuals, nbins=50, title="Residual Distribution", labels={"value": "Residual ($)"})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(x=y_pred_sample, y=residuals, title="Residuals vs Predicted Values",
                                 labels={"x": "Predicted Price ($)", "y": "Residual ($)"})
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Prediction Intervals")
            interval_df = pd.DataFrame({
                'Actual': y_test_sample, 'Predicted': y_pred_sample,
                'Lower Bound': lower_bound_sample, 'Upper Bound': upper_bound_sample
            })
            interval_df['Within Interval'] = (
                (interval_df['Actual'] >= interval_df['Lower Bound']) &
                (interval_df['Actual'] <= interval_df['Upper Bound'])
            )
            coverage = interval_df['Within Interval'].mean() * 100
            st.metric("Prediction Interval Coverage", f"{coverage:.2f}%")
            st.markdown("##### Sample Predictions with Intervals")
            sample_display = interval_df.sample(min(10, len(interval_df)))
            for col in ['Actual', 'Predicted', 'Lower Bound', 'Upper Bound']:
                sample_display[col] = sample_display[col].apply(lambda x: f"${x:,.0f}")
            st.dataframe(sample_display, hide_index=True)

    with tabs[5]:
        st.markdown('<div class="sub-header">💡 Advanced Insights</div>', unsafe_allow_html=True)
        st.markdown("#### Raw Data")
        st.dataframe(filtered_data, hide_index=True, height=400)
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        csv = convert_df(filtered_data)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="real_estate_data.csv",
            mime="text/csv",
        )
        st.markdown("#### Data Summary")
        summary = filtered_data.describe().T
        summary['count'] = summary['count'].astype(int)
        for col in ['mean', '50%', 'min', 'max']:
            if col in summary.columns:
                summary[col] = summary[col].apply(
                    lambda x: f"${x:,.0f}" if (isinstance(x, (int, float)) and x > 1000) else
                              f"{x:.2f}" if isinstance(x, (int, float)) else str(x)
                )
        st.dataframe(summary, height=300)
        numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            st.markdown("#### Correlation Matrix")
            corr = filtered_data[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        # Train models and store results
        if st.button("Train Models", type="primary"):
            with st.spinner('Training advanced prediction models...'):
                model_results, best_model_name, preprocessor, feature_importances, shap_values = train_advanced_models(filtered_data)
                if model_results:
                    st.session_state.model_results = model_results
                    st.session_state.best_model_name = best_model_name
                    st.session_state.preprocessor = preprocessor
                    st.session_state.feature_importances = feature_importances
                    st.session_state.shap_values = shap_values
                    st.success("Models trained successfully!")

else:
    st.markdown('<p class="error-message">Insufficient data (less than 50 properties). Please adjust filters.</p>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p style="color: #5c5c5c;">Powered by xAI | Built with Streamlit | Data updated as of June 21, 2025</p>
</div>
""", unsafe_allow_html=True)