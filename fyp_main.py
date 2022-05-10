# -*- coding: utf-8 -*-
"""FYP_MAIN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Yy83mwqighWNeZuswKoGH6d-vnSt9ZG6
"""

import pandas as pd
from fbprophet import Prophet
data = pd.read_csv('./train.csv')
data.head()
# data.shape



gData = data.groupby(['item', 'date'])['sales'].sum() #.agg(F.collect_list("SalesAmount")).sort('SalesItem')
gData = gData.to_frame().reset_index()
itemlist = gData.item.unique()

m = Prophet()

fcst_all = pd.DataFrame()  # store all forecasts here
for x in itemlist:
    temp = gData[gData.item == x]
    temp = temp.drop(columns=[ 'item'])
    temp['date'] = pd.to_datetime(temp['date'])
    temp = temp.set_index('date')

    d_df = temp.resample('MS').sum()
    d_df = d_df.reset_index().dropna()
    d_df.columns = ['ds', 'y']
    try:
        m = Prophet().fit(d_df)
        future = m.make_future_dataframe(periods=1, freq='MS')
    except ValueError:
        pass       

    fcst = m.predict(future)
    fcst['item'] = x  
    fcst['Fact'] = d_df['y'].reset_index(drop = True)
    fcst_all = pd.concat((fcst_all, fcst))

fcst_all.tail()

# fcst_all.drop(columns=['yhat_lower','yhat_upper','trend_lower','trend_upper','additive_terms','additive_terms_lower','additive_terms_upper','yearly','yearly_lower','yearly_upper','multiplicative_terms','multiplicative_terms_lower','multiplicative_terms_upper','Fact'], axis=1, inplace=True)

filtered_df = fcst_all.loc[(fcst_all['ds'] >= '2018-01-01')
                     & (fcst_all['ds'] <= '2018-01-01')]

filtered_df.tail()
filtered_df.to_csv('./finalpred.csv', index=False)