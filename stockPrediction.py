import streamlit
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.make_holidays import make_holidays_df
from plotly import graph_objs as graphObjects
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go

from thai_list_stock import thaiList

def prophet_model_performance(forecast):
    # Find the column containing the actual values (starting with 'y')
    actual_column = [col for col in forecast.columns if col.startswith('y')][0]
    
    # Extract actual and predicted values
    actual = forecast[actual_column].values[-14:]  # Last 14 actual values
    predicted = forecast['yhat'].values[-14:]  # Last 14 predicted values
    
    # Calculate error metrics
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean(np.square(actual - predicted))
    rmse = np.sqrt(mse)
    
    # Format and return error metrics
    error_metrics = {
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse
    }
    return error_metrics

TH_holidays = make_holidays_df(
    year_list=[2012 + i for i in range(10)], country='TH'
)

START_TIME = '2012-01-01'
END_TIME = date.today().strftime('%Y-%m-%d')

streamlit.title('Stock Forecast App')

stocks = thaiList()  # Adjusted ticker symbols
selected_stock = streamlit.selectbox('Select database for prediction', stocks)

nYears = streamlit.slider('Years of prediction:', 1, 4)
period = nYears * 365 # Assuming 365 days in a year

@streamlit.cache_data
@streamlit.cache_resource
def loadData(ticker):
    data = yf.download(ticker, interval='1d', period='max')
    data.reset_index(inplace=True)
    return data

data_load_state = streamlit.text('Loading Data...')
data = loadData(selected_stock)
data_load_state.text('Loading Data.. Done!')

streamlit.subheader('Raw Data')
streamlit.write(data.tail())

def plotRawData():
    figure = graphObjects.Figure()
    figure.add_trace(graphObjects.Scatter(x=data['Date'], y=data['Open'], name="Stock Open Prices"))
    figure.add_trace(graphObjects.Scatter(x=data['Date'], y=data['Close'], name="Stock Close Prices"))
    figure.add_trace(graphObjects.Scatter(x=data['Date'], y=data['High'], name="Stock High Prices"))
    figure.add_trace(graphObjects.Scatter(x=data['Date'], y=data['Low'], name="Stock Low Prices"))
    figure.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    streamlit.plotly_chart(figure)

plotRawData()

dataframeTrain = data[['Date', 'Close']]
dataframeTrain = dataframeTrain.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet(interval_width=0.9)
model.fit(dataframeTrain)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

streamlit.subheader('Forecast Data')
streamlit.write(forecast.tail())

streamlit.write(f'Forecast plot for {nYears} years')
figure1 = plot_plotly(model, forecast)
streamlit.plotly_chart(figure1)

streamlit.write("Forecast Components")
figure2 = model.plot_components(forecast)
streamlit.write(figure2)

current_price = data['Close'].iloc[-1]

def adjustFuturePrices(future: float):
    different = future - current_price
    different_scalar = different.item() if len(different) == 1 else different[0]
    return future - different_scalar*0

def predictSymbol(future: float):
    different = future - current_price
    different_scalar = different.item() if len(different) == 1 else different[0]
    if different_scalar > 0:
        return '▲'
    elif different_scalar < 0:
        return '▼'
    else:
        return '='


# Extend the period to 180 days (approximately 6 months)
period = 180

# Make future date predictions for the extended period
future_dates_extended = [date.today() + timedelta(days=i) for i in range(1, period + 1)]
future_dataframe_extended = pd.DataFrame({'ds': future_dates_extended})
future_forecast_extended = model.predict(future_dataframe_extended)

# Select the predictions for the extended period
future_prices_extended = future_forecast_extended[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

changePercent = ((future_prices_extended['yhat'][0] - current_price)/current_price) * 100

# Plot the predicted prices along with uncertainty intervals for the extended period
fig = go.Figure()

# Add the predicted prices
fig.add_trace(go.Scatter(x=future_prices_extended['ds'], y=adjustFuturePrices(future_prices_extended['yhat']), mode='lines', name='Predicted Price'))

# Add the uncertainty intervals
fig.add_trace(go.Scatter(
    x=future_prices_extended['ds'],
    y=adjustFuturePrices(future_prices_extended['yhat_upper']),
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=future_prices_extended['ds'],
    y=adjustFuturePrices(future_prices_extended['yhat_lower']),
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(0,100,80,0.2)',
    line=dict(width=0),
    name='Uncertainty'
))

# Customize layout
fig.update_layout(
    title='Predicted Prices for the Next {} Months | Current Prices: {:.2f} {} (change % in next 1 day: {:.2f}%)'.format(int(period/30), current_price, predictSymbol(future_prices_extended['yhat']), changePercent),
    xaxis_title='Date',
    yaxis_title='Price',
    hovermode='x unified'
)

# Show the plot
streamlit.write(fig)

# Calculate and show error metrics
streamlit.write("Error Metrics:")
error_metrics = prophet_model_performance(forecast)
streamlit.write(error_metrics)
