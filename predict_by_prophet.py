import pandas as pd
from fbprophet import Prophet

if __name__ == '__main__':
    sales_df = pd.read_csv('training_data.csv')
    print("-----------Stock market price of MSFT on daily basis-------------------")
    print(sales_df)
    print(sales_df, sales_df.columns,sales_df.index)
    sales_df = sales_df.rename(columns={'index': 'ds', 'close': 'y'})
    model = Prophet(changepoint_range=1) #instantiate Prophet
    model.fit(sales_df); #fit the model with your dataframe
    future_dates = model.make_future_dataframe(periods=16, freq = 'd')
    forecast_data = model.predict(future_dates).rename(columns={'ds': 'Date', 'yhat': 'Closing Price'})
    print("-----------Prediction of stock price on daily basis-------------------")
    print(forecast_data[['Date', 'Closing Price']][4634:])
