
import pandas as pd
from fbprophet import Prophet
import streamlit as st
import streamlit.components.v1 as stc
import datetime
st.header('NIK Productions')
st.subheader('Future Sales Prediction')


data_file = st.file_uploader("Upload CSV", type=["csv"])
today = datetime.date.today()
d = st.date_input(
     "Select Next Month for prediction",today)
st.write('Month is:', d)


if data_file is not None:
    file_details = {"filename": data_file.name, "filetype": data_file.type,
                    "filesize": data_file.size}
    st.write(file_details)
    data = pd.read_csv(data_file)
    st.dataframe(data)
    data.head()
    # data.shape
    gData = data.groupby(['item', 'date'])['sales'].sum()  # .agg(F.collect_list("SalesAmount")).sort('SalesItem')
    gData = gData.to_frame().reset_index()
    itemlist = gData.item.unique()

    m = Prophet()

    fcst_all = pd.DataFrame()  # store all forecasts here
    for x in itemlist:
        temp = gData[gData.item == x]
        temp = temp.drop(columns=['item'])
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
        fcst['Fact'] = d_df['y'].reset_index(drop=True)
        fcst_all = pd.concat((fcst_all, fcst))
    fcst_all.tail()

    fcst_all.drop(columns=['yhat_lower','yhat_upper','trend_lower','trend_upper','additive_terms','additive_terms_lower','additive_terms_upper','yearly','yearly_lower','yearly_upper','multiplicative_terms','multiplicative_terms_lower','multiplicative_terms_upper','Fact'], axis=1, inplace=True)
     da = pd.to_datetime(d)
    filtered_df = fcst_all.loc[(fcst_all['ds'] >= da) & (fcst_all['ds'] <= da)]

    filtered_df.tail()
    st.table(filtered_df)
    filtered_df.to_csv('./finalpred.csv', index=False)

