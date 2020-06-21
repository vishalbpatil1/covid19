import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings 
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima_model import ARIMA

st.title('Covid-19 Data Analysis In India')
st.header('covid19 ')

data2=pd.read_csv('data\\covid_19_india.csv')
states=data2['State/UnionTerritory'].unique()
s=st.selectbox('select state',states.tolist())
data_all=data2[['State/UnionTerritory','Date','Cured','Confirmed','Deaths']]
data=data_all[data_all['State/UnionTerritory']==s]
#data=data[['State/UnionTerritory','Date','Cured','Confirmed','Deaths']]
#data_all=data2[['State/UnionTerritory','Date','Cured','Deaths','Confirmed']]
st.write(data.tail())
st.header('covid -19  data visualization ')
fig = px.bar(data, y='Confirmed', x='Date',color='Confirmed')
fig.update_layout(title_text='Covid-19 cummulative confirmed cases in  ' +str(s))
st.plotly_chart(fig, use_container_width=True )

fig = px.bar(data, y='Deaths', x='Date',color='Deaths')
fig.update_layout(title_text='Covid-19 cummulative Death  cases in ' +str(s))
st.plotly_chart(fig, use_container_width=True )




fig = go.Figure()
# Create and style traces
fig.add_trace(go.Scatter(x=data['Date'], y=data['Confirmed'], name='Confirmed'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Cured'], name = 'Cured'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Deaths'], name='Death '))
fig.add_trace(go.Scatter(x=['25/03/20','25/03/20'], y=[0,np.max(data['Confirmed'])],text='all india lockdwon',mode='lines', name='phase 1'))
fig.add_annotation(x='25/03/20',y=np.max(data['Confirmed']-200),text="First Lockdwon ( All India)")

fig.add_trace(go.Scatter(x=['15/04/20','15/04/20'], y=[0,np.max(data['Confirmed'])],text='all india lockdwon',mode='lines', name='phase 2'))
fig.add_annotation(x='15/04/20',y=np.max(data['Confirmed']-400),text="Second Lockdwon ( All India)")

fig.add_trace(go.Scatter(x=['04/05/20','04/05/20'], y=[0,np.max(data['Confirmed'])],text='all india lockdwon',mode='lines', name='phase 3'))
fig.add_annotation(x='04/05/20',y=np.max(data['Confirmed']-600),text="Third Lockdwon ( All India)")

fig.add_trace(go.Scatter(x=['18/05/20','18/05/20'], y=[0,np.max(data['Confirmed'])],text='all india lockdwon',mode='lines', name='phase 4'))
fig.add_annotation(x='18/05/20',y=np.max(data['Confirmed']-800),text="Fourth Lockdwon ( All India)")

fig.add_trace(go.Scatter(x=['01/06/20','01/06/20'], y=[0,np.max(data['Confirmed'])],text='all india lockdwon',mode='lines', name='phase 5'))
fig.add_annotation(x='01/06/20',y=np.max(data['Confirmed']-1000),text="Fifth Lockdwon ( containment zones)")

fig.update_layout(title_text='Covid-19 cases tred in '+str(s))

st.plotly_chart(fig, use_container_width=True )


# time series forecastiing 
st.title('Time Series forecasting for next 5 day')
data_timeseries=pd.DataFrame()
dd=data[data['State/UnionTerritory']==s].iloc[-1]['Date']
data_timeseries['Confirmed_cases']=data[data['State/UnionTerritory']==s]['Confirmed']
date=pd.to_datetime(data[data['State/UnionTerritory']==s]['Date'])
data_timeseries.index=date
#st.write(data_timeseries)
#import numpy as np
r=[]
b=[]
order=[]
for i in range(3):
    for j in range(2):
        for k in range(3):
            try:
                model = ARIMA(np.log(data_timeseries), order=(i,j,k))
                model_fit = model.fit(disp=False)
                r.append(sum(model_fit.resid))
                a=(i,j,k)
                order.append(a)
                b.append(sum((model_fit.resid)**2))
                #print(' ARIMA ',a,'MSE ',sum((model_fit.resid)**2))
            except ValueError:
                continue
model = ARIMA(np.log(data_timeseries), order=order[b.index(np.min(b))])
model_fit = model.fit(disp=False)
forecast=model_fit.forecast(steps=5)
f=pd.DataFrame()
f['Date']=pd.date_range(start=dd,periods=5)
f['forecast']=np.exp(forecast[0]).astype(int)
f['lower_limit']=np.exp(forecast[2][:,0]).astype(int)
f['upper_limit']=np.exp(forecast[2][:,1]).astype(int)             
fig = go.Figure(data=[go.Table(
    header=dict(values=list(f.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[f['Date'],f['forecast'],f['lower_limit'],f['upper_limit']],
               fill_color='lavender',
               align='left'))
])
st.plotly_chart(fig)
#
#
fig=px.bar(data_all, y="State/UnionTerritory", x="Confirmed", color='Confirmed',animation_frame="Date")
fig.update_layout(title_text='Covid-19 cases tred india')
st.plotly_chart(fig, use_container_width=True )
#
#

st.title('Datewise presentation of Covid-19')
date=data_all['Date'].unique()
d=st.selectbox('select date',date.tolist())
total_data=data_all[data_all['Date']==d].sort_values(by='Confirmed',ascending=False)
#total_data=total_data.sort_values(by='Confirmed',ascending=False)
total_data['Active']=total_data['Confirmed']-(total_data['Deaths']+total_data['Cured'])
total_data['Death Rate (per 100)']=np.round(100*total_data["Deaths"]/total_data["Confirmed"],2)
total_data["Cure Rate (per 100)"] = np.round(100*total_data["Cured"]/total_data["Confirmed"],2)

fig = px.bar(total_data, y='Confirmed',x='State/UnionTerritory')
fig.update_layout(title_text='{ Covid-19 Total Confirmed cases in India } Date-' +str(d))
st.plotly_chart(fig, use_container_width=True )

fig = go.Figure(data=[go.Table(
    header=dict(values=list(total_data.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[total_data['State/UnionTerritory'],total_data['Date'],total_data['Cured'],total_data['Confirmed'],total_data['Active'],total_data['Deaths'],total_data['Death Rate (per 100)'],total_data['Cure Rate (per 100)']],
               fill_color='lavender',
               align='left'))
])
st.plotly_chart(fig, use_container_width=True )

