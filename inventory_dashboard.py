"""
Author: Jaydin Freeman
Date: 03-12-2024
Description: This is a Streamlit dashboard for inventory management. It allows users to visualize demand data, 
forecast future demand, and view key performance indicators (KPIs) for grocery stores.

"""

import ray
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error


@ray.remote
def triple_exponential_smoothing(series_param, slen_param, alpha_param, beta_param, gamma_param, n_preds_param):
    """
    Applies triple exponential smoothing to a time series.

    Parameters:
    - series (list): The input time series data.
    - slen_param (int): The length of the seasonal component.
    - alpha_param (float): The smoothing factor for the level component.
    - beta_param (float): The smoothing factor for the trend component.
    - gamma_param (float): The smoothing factor for the seasonal component.
    - n_preds_param (int): The number of future predictions to make.

    Returns:
    - result (list): The smoothed time series data with future predictions.
    """
    result = []
    seasonals = initial_seasonal_components(series_param, slen_param)
    for i in range(len(series_param)+n_preds_param):
        if i == 0:  # initial values
            smooth = series_param[0]
            trend = initial_trend(series_param, slen_param)
            result.append(series_param[0])
            continue
        if i >= len(series_param):  # forecasting
            m = i - len(series_param) + 1
            result.append((smooth + m*trend) + seasonals[i % slen_param])
        else:
            val = series_param[i]
            last_smooth, smooth = smooth, alpha_param * \
                (val-seasonals[i % slen_param]) + \
                (1-alpha_param)*(smooth+trend)
            trend = beta_param * (smooth-last_smooth) + (1-beta_param)*trend
            seasonals[i % slen_param] = gamma_param * \
                (val-smooth) + (1-gamma_param)*seasonals[i % slen_param]
            result.append(smooth+trend+seasonals[i % slen_param])
    return result


def grid_search_tes(data_param, slen_range_param, alpha_range_param, beta_range_param, gamma_range_param, n_preds_param):
    """
    Perform a grid search to find the best parameters for triple exponential smoothing.

    Parameters:
    - data_param (list): The input data for forecasting.
    - slen_range_param (list): The range of values for the seasonal length parameter.
    - alpha_range_param (list): The range of values for the alpha parameter.
    - beta_range_param (list): The range of values for the beta parameter.
    - gamma_range_param (list): The range of values for the gamma parameter.
    - n_preds_param (int): The number of predictions to make.

    Returns:
    - best_params (tuple): The best parameters found during the grid search.
    """
    best_params_inner = None
    best_mape = float('inf')

    for s in slen_range_param:
        for a in alpha_range_param:
            for b in beta_range_param:
                for g in gamma_range_param:
                    f = ray.get(triple_exponential_smoothing.remote(
                        data_param, s, a, b, g, n_preds_param))
                    m = calculate_forecast_accuracy(
                        data_param[-n_preds_param:], f[-n_preds_param:])[3]
                    if m < best_mape:
                        best_mape = m
                        best_params_inner = (s, a, b, g)

    return best_params_inner


def calculate_forecast_accuracy(actual_param, forecast_param):
    """
    Calculates the forecast accuracy metrics for a given set of actual and forecasted values.

    Parameters:
    actual_param (array-like): The actual values.
    forecast_param (array-like): The forecasted values.

    Returns:
    tuple: A tuple containing the following forecast accuracy metrics:
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Mean Absolute Percentage Error (MAPE)
    """
    mae_inner = mean_absolute_error(actual_param, forecast_param)
    mse_inner = mean_squared_error(actual_param, forecast_param)
    rmse_inner = np.sqrt(mse_inner)
    mape_inner = np.mean(
        np.abs((actual_param - forecast_param) / actual_param)) * 100
    return mae_inner, mse_inner, rmse_inner, mape_inner


def initial_seasonal_components(series_param, slen_param):
    """
    Compute the initial seasonal components for a given time series.

    Args:
        series_param (list): The time series data.
        slen_param (int): The length of each season.

    Returns:
        dict: A dictionary containing the initial seasonal components.

    """
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series_param)/slen_param)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(
            sum(series_param[slen_param*j:slen_param*j+slen_param])/float(slen_param))
    # initial values
    for i in range(slen_param):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series_param[slen_param *
                                                 j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals


def initial_trend(series_param, slen_param):
    """
    Calculate the initial trend of a time series.

    Parameters:
    - series_param (list): The time series data.
    - slen_param (int): The length of the seasonal pattern.

    Returns:
    - float: The initial trend value.
    """
    total = 0.0
    for i in range(slen_param):
        total += (series_param[i+slen_param] - series_param[i]) / slen_param
    return total / slen_param


def load_data():
    # Placeholder for data loading
    """
    Load data for the inventory dashboard.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded data with columns "Date" and "Demand".
    """
    dates_inner = pd.date_range(start="2021-01-01", end="2021-12-31", freq="D")
    data_inner = np.random.randint(10, 100, size=len(dates_inner))
    return pd.DataFrame({"Date": dates_inner, "Demand": data_inner})


def load_inventory_data():
    # Placeholder for inventory data loading
    """
    Load inventory data.

    This function generates random inventory data for each day within a specified date range.

    Returns:
        pandas.DataFrame: A DataFrame containing the inventory data with columns "Date" and "Inventory".
    """
    dates_inner = pd.date_range(start="2021-01-01", end="2021-12-31", freq="D")
    data_inner = np.random.randint(50, 200, size=len(dates_inner))
    return pd.DataFrame({"Date": dates_inner, "Inventory": data_inner})


def plot_supply_demand_forecast(dates_param, original_param, forecast_param, inventory_data_param, demand_data_param):
    fig = go.Figure()

    # Actual Demand
    fig.add_trace(go.Scatter(
        x=demand_data_param['Date'], y=demand_data_param['Demand'], name='Actual Demand'))

    # Forecasted Demand
    fig.add_trace(go.Scatter(x=dates_param, y=forecast_param, name='Forecasted Demand'))

    # Inventory (Supply)
    fig.add_trace(go.Scatter(
        x=inventory_data_param['Date'], y=inventory_data_param['Inventory'], name='Inventory (Supply)'))

    # Update layout
    fig.update_layout(title='Supply, Demand, and Forecast',
                      xaxis_title='Date',
                      yaxis_title='Units',
                      xaxis=dict(
                          rangeselector=dict(
                              buttons=list([
                                  dict(count=7, label="1w", step="day",
                                       stepmode="backward"),
                                  dict(count=1, label="1m", step="month",
                                       stepmode="backward"),
                                  dict(count=3, label="3m", step="month",
                                       stepmode="backward"),
                                  dict(count=6, label="6m", step="month",
                                       stepmode="backward"),
                                  dict(count=1, label="YTD",
                                       step="year", stepmode="todate"),
                                  dict(count=1, label="1y", step="year",
                                       stepmode="backward"),
                                  dict(step="all")
                              ])
                          ),
                          rangeslider=dict(visible=True),
                          type="date"
                      ))

    st.plotly_chart(fig)


st.title('Inventory Management Dashboard')

data_load_state = st.text('Loading data...')
data = load_data()
inventory_data = load_inventory_data()
data_load_state.text("Data loaded!")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

# Forecasting Section
st.subheader('Forecast Demand')
st.write('This section allows you to forecast future demand using various methods. The graph will display the original demand data and the forecasted values.')

forecasting_method = st.selectbox(
    'Select forecasting method:',
    ['AI/ML', 'Triple Exponential Smoothing',
        'Reinforcement Learning', 'Wagner-Whitin']
)

if forecasting_method == 'Triple Exponential Smoothing':
    st.write('Triple exponential smoothing is a forecasting method that considers seasonality, trend, and level. It uses three smoothing parameters (alpha, beta, gamma) to update the estimates for each component at each time step.')

    slen = st.slider('Seasonal length (number of periods in a season):',
                     min_value=2, max_value=12, value=6)
    alpha = st.slider('Alpha (level smoothing parameter):',
                      min_value=0.0, max_value=1.0, value=0.2)
    beta = st.slider('Beta (trend smoothing parameter):',
                     min_value=0.0, max_value=1.0, value=0.1)
    gamma = st.slider('Gamma (seasonality smoothing parameter):',
                      min_value=0.0, max_value=1.0, value=0.1)
    n_preds = st.number_input(
        'Number of periods to forecast:', min_value=1, max_value=100, value=30)

    if st.button('Forecast'):
        forecast = ray.get(triple_exponential_smoothing.remote(
            data['Demand'], slen, alpha, beta, gamma, n_preds))
        dates = pd.date_range(
            start=data['Date'].iloc[0], periods=len(data)+n_preds, freq='D')
        plot_supply_demand_forecast(dates, data['Demand'].tolist(
        ) + [None]*n_preds, forecast, inventory_data, data)

    if st.button('Grid Search'):
        slen_range = range(2, 13)
        alpha_range = np.arange(0.1, 1.0, 0.5)
        beta_range = np.arange(0.1, 1.0, 0.05)
        gamma_range = np.arange(0.1, 1.0, 0.05)
        best_params = grid_search_tes(
            data['Demand'], slen_range, alpha_range, beta_range, gamma_range, n_preds)
        st.write(
            f"Best parameters: slen={best_params[0]}, alpha={best_params[1]:.1f}, beta={best_params[2]:.1f}, gamma={best_params[3]:.1f}")

        forecast = ray.get(triple_exponential_smoothing.remote(
            data['Demand'], *best_params, n_preds))
        dates = pd.date_range(
            start=data['Date'].iloc[0], periods=len(data)+n_preds, freq='D')
        plot_supply_demand_forecast(dates, data['Demand'].tolist(
        ) + [None]*n_preds, forecast, inventory_data, data)

# Placeholder sections for AI/ML and Reinforcement Learning forecasting
# Should be replaced with actual model prediction and plotting calls
elif forecasting_method == 'AI/ML':
    st.write('AI/ML forecasting not implemented.')
    # forecast = predict_demand_ml(data['Demand'])
elif forecasting_method == 'Reinforcement Learning':
    st.write('Reinforcement Learning forecasting not implemented.')
    # forecast = predict_demand_rl(data['Demand'])

st.subheader('Forecast Performance Score Card')
if 'forecast' in locals():
    mae, mse, rmse, mape = calculate_forecast_accuracy(
        data['Demand'][-n_preds:], forecast[-n_preds:])
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
else:
    st.write(
        "No forecast available. Please generate a forecast to see performance metrics.")

st.subheader('Best Practice KPIs for Grocery Stores')
kpis = {
    'Inventory Turnover Ratio': 'Measures how quickly inventory is sold and replaced. Higher is better.',
    'Out-of-Stock Percentage': 'Percentage of items that are out of stock. Lower is better.',
    'Waste Percentage': 'Percentage of inventory that goes to waste. Lower is better.',
    'Service Level': 'Percentage of demand that is met from available inventory. Higher is better.',
    'Forecast Accuracy (MAPE)': 'Mean Absolute Percentage Error of the demand forecast. Lower is better.'
}

for kpi, description in kpis.items():
    st.write(f"- {kpi}: {description}")
