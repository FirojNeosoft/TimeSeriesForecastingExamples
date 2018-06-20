import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
# from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    df = pd.read_csv('training_data.csv')
    print("-----------Stock market price of MSFT on daily basis-------------------")
    print(df)
    model = ARIMA(df['close'], order=(5, 1, 0))  #instantiate ARIMA model
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=16)
    history_dates = pd.to_datetime(df['index']).sort_values()
    last_date = history_dates.max()
    dates = pd.date_range(
        start=last_date+ dt.timedelta(days=1),
        periods=16,  # An extra in case we include start
        freq='D')
    odf = pd.DataFrame({'Date':dates, 'Close': output[0]})
    print("-----------Prediction of stock price on daily basis-------------------")
    print(odf)
    # odf.Date = pd.to_datetime(odf['Date'], format='%Y-%m-%d %H:%M:%S.%f')
    # odf.set_index(['Date'], inplace=True)
    # odf.plot(kind='line', color='red')

    odf = odf.sort_values('Date', ascending=True)
    plt.plot(odf['Date'], odf['Close'])
    plt.title("Prediction of stock price of MSFT")
    plt.show()