# -*- coding: utf-8 -*-
"""FYP_All_Items.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HjI-pfb95kuiR8ijaB7GBYcf3d1ogtPl
"""

import pandas as pd

df = pd.read_csv('retail_sales.csv');
# df.drop(columns=['item','store'], axis=1, inplace=True)
df.head()

# Commented out IPython magic to ensure Python compatibility.
import fbprophet
import matplotlib.pyplot as plt
# %matplotlib inline

df.plot()
df.head()

df.columns = ['ds','y']
df.head()

# df.set_index('ds', inplace=True)
# df.index = pd.to_datetime(df.index)
# df.resample('1M').mean()

df['ds'] = pd.to_datetime(df['ds'])
df

from fbprophet import Prophet

dir(Prophet)

model = Prophet()

df.columns

model.fit(df)

model.component_modes

df.tail()

future_dates = model.make_future_dataframe(periods=1, freq='MS')

future_dates

prediction = model.predict(future_dates)

prediction.head()

model.plot(prediction)

model.plot_components(prediction)

from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(model,initial="30 days", period="30 days", horizon="30 days")

df_cv.tail()

from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.head()

from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv,metric="rmse")