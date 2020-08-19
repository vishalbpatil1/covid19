import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

import warnings 
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima_model import ARIMA

# Title
st.title('Coronavirus (COVID-19) statistics')
st.header('The COVID-19 pandemic in India is part of the worldwide pandemic of coronavirus disease 2019 (COVID-19) caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The first case of COVID-19 in India, which originated from China, was reported on 30 January 2020. India currently has the largest number of confirmed cases in Asia, and has the third highest number of confirmed cases in the world after the United States and Brazil with the number of total confirmed cases breaching the 100,000 mark on 19 May, 200,000 on 3 June. 1,000,000 confirmed cases on 17 July 2020.')
covid_img= Image.open('data\\img.png')
st.image(covid_img)
data2=pd.read_csv('data\\covid_19_india.csv')
data_state=pd.read_csv('data\\StatewiseTestingDetails.csv')
data_state=data_state.fillna(0)
data_state['State'].unique().tolist()



states=data2['State/UnionTerritory'].unique()
s=st.sidebar.selectbox('select state',states.tolist())
data_all=data2[['State/UnionTerritory','Date','Cured','Confirmed','Deaths']]
data=data_all[data_all['State/UnionTerritory']==s]
data1=data_state[data_state['State']==s]
#data=data[['State/UnionTerritory','Date','Cured','Confirmed','Deaths']]
#data_all=data2[['State/UnionTerritory','Date','Cured','Deaths','Confirmed']]
#st.write(data.tail())

# graph_1
st.title('covid -19 Cases data visualization ')
fig = px.bar(data1, y='TotalSamples', x='Date',
             labels={'TotalSamples':'Total No Of Sample collected','Date':'Date / Day'}, height=400)
fig.update_layout(title_text='Total Sample collected in ' +str(s))
st.plotly_chart(fig, use_container_width=True )

# graph_2
fig = px.bar(data, y='Confirmed', x='Date',color='Confirmed')
fig.update_layout(title_text='Covid-19 cummulative confirmed cases in  ' +str(s))
st.plotly_chart(fig, use_container_width=True )

# graph_3
fig = px.bar(data, y='Deaths', x='Date',color='Deaths')
fig.update_layout(title_text='Covid-19 cummulative Death  cases in ' +str(s))
st.plotly_chart(fig, use_container_width=True )



# graph_4
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Confirmed'], name='Confirmed'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Cured'], name = 'Cured'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Deaths'], name='Death '))
fig.add_trace(go.Scatter(x=['25/03/20','25/03/20'], y=[0,np.max(data['Confirmed'])],text='First Lockdwon ( All India)',mode='lines', name='phase 1'))
fig.add_annotation(x='25/03/20',y=np.max(data['Confirmed']-200),text="P1")

fig.add_trace(go.Scatter(x=['15/04/20','15/04/20'], y=[0,np.max(data['Confirmed'])],text='Second Lockdwon ( All India)',mode='lines', name='phase 2'))
fig.add_annotation(x='15/04/20',y=np.max(data['Confirmed']-400),text="P2")

fig.add_trace(go.Scatter(x=['04/05/20','04/05/20'], y=[0,np.max(data['Confirmed'])],text='Third Lockdwon ( All India)"',mode='lines', name='phase 3'))
fig.add_annotation(x='04/05/20',y=np.max(data['Confirmed']-600),text="P3")

fig.add_trace(go.Scatter(x=['18/05/20','18/05/20'], y=[0,np.max(data['Confirmed'])],text='Fourth Lockdwon ( All India)',mode='lines', name='phase 4'))
fig.add_annotation(x='18/05/20',y=np.max(data['Confirmed']-800),text="P4")

fig.add_trace(go.Scatter(x=['01/06/20','01/06/20'], y=[0,np.max(data['Confirmed'])],text='Fifth Lockdwon ( containment zones)',mode='lines', name='phase 5'))
fig.add_annotation(x='01/06/20',y=np.max(data['Confirmed']-1000),text="P5")

fig.update_layout(title_text='Covid-19 Cases (Confirmed cases,Cured cases,Death cases) in '+str(s))
st.plotly_chart(fig, use_container_width=True )


# graph_5
fig = go.Figure(data=[
    go.Bar(name='Total sample', x=data1['Date'], y=data1['TotalSamples']),
    go.Bar(name='Positive cases ', x=data1['Date'], y=data1['Positive']),
    go.Bar(name='Negative cases', x=data1['Date'], y=data1['Negative'])
])
# Change the bar mode
fig.update_layout(barmode='stack')
fig.update_layout(title_text='Covid-19 Date-wise Positive and Negative Cases in '+str(s))
st.plotly_chart(fig, use_container_width=True )


# time series forecastiing 
data_timeseries=pd.DataFrame()
dd=data[data['State/UnionTerritory']==s].iloc[-1]['Date']
data_timeseries['Confirmed_cases']=data[data['State/UnionTerritory']==s]['Confirmed']
date=pd.to_datetime(data[data['State/UnionTerritory']==s]['Date'])
data_timeseries.index=date

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

# graph_7           
fig = go.Figure(data=[go.Table(
    header=dict(values=list(f.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[f['Date'],f['forecast'],f['lower_limit'],f['upper_limit']],
               fill_color='lavender',
               align='left'))
])
fig.update_layout(title_text='Time Series forecasting for Coronavirus positive cases -- '+str(s))
st.plotly_chart(fig)
#
#
#fig=px.bar(data_all, y="State/UnionTerritory", x="Confirmed", color='Confirmed',animation_frame="Date")
#fig.update_layout(title_text='Covid-19 cases tred india')
#st.plotly_chart(fig, use_container_width=True )
#
#

st.title('Date-wise Presentation Of Covid-19 (Coronavirus)')
date=data_all['Date'].unique()
d=st.sidebar.selectbox('select date',date.tolist())
total_data=data_all[data_all['Date']==d].sort_values(by='Confirmed',ascending=False)
#total_data=total_data.sort_values(by='Confirmed',ascending=False)
total_data['Active']=total_data['Confirmed']-(total_data['Deaths']+total_data['Cured'])
total_data['Death Rate (per 100)']=np.round(100*total_data["Deaths"]/total_data["Confirmed"],2)
total_data["Cure Rate (per 100)"] = np.round(100*total_data["Cured"]/total_data["Confirmed"],2)

fig = px.bar(total_data, y='Confirmed',x='State/UnionTerritory')
fig.update_layout(title_text='{ Covid-19 Sate-wise Total Confirmed cases in India } Date-' +str(d))
st.plotly_chart(fig, use_container_width=True )


# Graph_8
fig = go.Figure(data=[go.Table(
    header=dict(values=list(total_data.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[total_data['State/UnionTerritory'],total_data['Date'],total_data['Cured'],total_data['Confirmed'],total_data['Active'],total_data['Deaths'],total_data['Death Rate (per 100)'],total_data['Cure Rate (per 100)']],
               fill_color='lavender',
               align='left'))
])
fig.update_layout(title_text='{ State-wise Cured rate and Death rate } Date-' +str(d))
st.plotly_chart(fig, use_container_width=True )



date_=data_state['Date'].unique()
d_='20'+d[6:8]+'-'+d[3:5]+'-'+d[0:2]
dd=data_state[data_state['Date']==d_]



fig = px.pie(dd, values=dd['TotalSamples'], names=dd['State'], title='Percentage Of Total Sample testing--Date '+str(d))
st.plotly_chart(fig, use_container_width=True )

fig = px.pie(dd, values=dd['Positive'], names=dd['State'], title=' Percentage Of Covid 19 Positive Cases --Date'+str(d))
st.plotly_chart(fig, use_container_width=True )



ddd=dd.sort_values('Positive',ascending=False).head(5)
labels =ddd['State']


# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
fig.add_trace(go.Pie(labels=labels, values=ddd['TotalSamples'], name="Total samples"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=ddd['Negative'], name="Negative"),
              1, 2)
fig.add_trace(go.Pie(labels=labels, values=ddd['Positive'], name="Positive"),
              1, 3)
# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")
fig.update_layout(
    title_text="Top 5 State Where highest spread of covid-19 (Coronavirus)--Date-"+str(d),
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Samples', x=0.09, y=0.5, font_size=10, showarrow=False),
                 dict(text='Negative', x=0.5, y=0.5, font_size=10, showarrow=False),
                dict(text='Positive', x=0.92, y=0.5, font_size=10, showarrow=False)])

st.plotly_chart(fig, use_container_width=True )

