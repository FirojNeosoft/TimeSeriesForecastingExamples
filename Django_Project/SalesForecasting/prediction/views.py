import json
import datetime as dt
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from statsmodels.tsa.arima_model import ARIMA
from django.shortcuts import render
from django.views.generic import View


class TrainModel(View):
    """
    Train model
    """

    def get(self, request):
        return render(request, "training.html")

    def post(self, request):
        df = pd.read_csv(request.FILES['dataset_file'])
        print("-----------Stock market price of MSFT on daily basis-------------------")
        print(df)
        model = ARIMA(df['close'], order=(5, 1, 0))  # instantiate ARIMA model
        model_fit = model.fit(disp=0)
        df.to_csv(settings.TRAINING_FILE_PATH)
        return render(request, 'training.html', {'msg': 'Successfully model trained.'})


class Prediction(View):
    """
    Prediction
    """

    def get(self, request):
        return render(request, "prediction.html")

    def post(self, request):
        days_count = int(request.POST['days_count'])
        df = pd.read_csv(settings.TRAINING_FILE_PATH)
        print("-----------Stock market price of MSFT on daily basis-------------------")
        print(days_count)
        model = ARIMA(df['close'], order=(5, 1, 0))  # instantiate ARIMA model
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(steps=days_count)
        history_dates = pd.to_datetime(df['index']).sort_values()
        last_date = history_dates.max()
        dates = pd.date_range(
            start=last_date + dt.timedelta(days=1),
            periods=days_count,  # An extra in case we include start
            freq='D')
        print("output=", output[0], dates)
        # odf = pd.DataFrame({'Date': dates, 'Close': output[0]})
        # print("-----------Prediction of stock price on daily basis-------------------")
        # print(odf)
        return JsonResponse({'Date': [x for x in list(dates.format('%m/%d/%Y')) if x], 'Close': list(output[0])})
        # return render(request, 'prediction.html', {'msg': 'Successfully model trained.'})

