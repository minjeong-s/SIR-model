# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# data를 최신으로 업데이트 하고 싶으면
# enddate를 두개를 바꿔줘야함

import pandas as pd
import numpy as np
import requests 
import seaborn as sns
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup

key='6HDS7N5bBf5gnRNO3redozFB9vHRmZlwEXAsDnn%2Fzb%2F1xMLiVjm4lH3waenlF3hLoTO4Pf3Q%2BKONpIDZIFpkMQ%3D%3D'
startdate='20200204'
enddate='20210721'
url='http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson?startCreateDt='+startdate+'&endCreateDt='+enddate+'&serviceKey='+key

res=requests.get(url)
res

soup=BeautifulSoup(res.content, 'html.parser')
soup

# _____________________

keylist=['statedt','decidecnt','deathcnt']
for key in keylist:
    print(len(soup.find_all(key)))

# +
keylist=['statedt','decidecnt','deathcnt']

data=[]

for i in np.arange(0,len(soup.find_all('statedt'))):
    temp=[]
    for key in keylist:
        temp.append(soup.find_all(key)[i].text)
    data.append(temp)

print(data) 
# -

mydf=pd.DataFrame(data)
mydf.columns=['statedt','decidecnt','deathcnt']
mydf=mydf.sort_values(by='statedt').reset_index(drop=True)
mydf

key='6HDS7N5bBf5gnRNO3redozFB9vHRmZlwEXAsDnn%2Fzb%2F1xMLiVjm4lH3waenlF3hLoTO4Pf3Q%2BKONpIDZIFpkMQ%3D%3D'
startdate='20200303'
enddate='20210721'
url='http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson?startCreateDt='+startdate+'&endCreateDt='+enddate+'&serviceKey='+key


res=requests.get(url)
res


soup=BeautifulSoup(res.content, 'html.parser')
soup


keylist=['statedt', 'carecnt']
for key in keylist:
    print(len(soup.find_all(key)))


# +

carecnt=[]

for i in np.arange(0,len(soup.find_all('statedt'))):
    temp=[]
    for key in keylist:
        temp.append(soup.find_all(key)[i].text)
    carecnt.append(temp)

print(carecnt)

# +
 
temp=pd.DataFrame(carecnt)
temp.columns=['statedt','carecnt']
temp=temp.sort_values(by='statedt').reset_index(drop=True)
temp['carecnt']=temp['carecnt'].astype(int)
temp

# -

raw=pd.merge(mydf, temp, how='outer', on='statedt')

# +
raw.to_csv('rawdata2.csv') 

'''
20.02.05~21.07.08까지 공공데이터 포털 보건복지부 코로나19 감염현황 데이터의
데이터 항목 중에서
decidecnt, carecnt, clearcnt, deathcnt 를 그대로 가져와서 그대로 저장한 것
가공하지 않았음
'''

# -

# _______________________________

# +
# 여기서부터 가공 시작 - 7일평균, 누적일일로 바꾸고 그런 것들..  
# -

df = pd.read_csv('rawdata2.csv')

df['cumdeath']=df.deathcnt

df['cumclear']=df.clearcnt

df['cumdecide']=df.decidecnt

df

# 누적을 daily로 바꾸기 
df[['decidecnt', 'clearcnt', 'deathcnt']]=df[['decidecnt', 'clearcnt', 'deathcnt']].astype('int').diff()

df['Date']=pd.to_datetime(df['statedt'].astype(str), format='%Y%m%d')

df=df.set_index('Date')
df.iloc[0,2]=19
df.iloc[0,3]=1
df.iloc[0,4]=0

df=df.drop(['Unnamed: 0'],axis=1)
df

df.columns=['statedt', 'newly_confirmed', 'daily_clear', 'daily_death', 'Quarantined', 'cumdeath' ,'cumclear', 'cumdecide']

df

# _________________________________________

# 7 days moving average 
df['Quarantined_ma']=df['Quarantined'].rolling("7d").mean()
df['newly_confirmed_ma']=df['newly_confirmed'].rolling("7d").mean()
df['daily_clear_ma']=df['daily_clear'].rolling("7d").mean()
df['daily_death_ma']=df['daily_death'].rolling("7d").mean()

df

df.to_csv('mydata2.csv')

df['daily_death_ma'].plot()

# +
data = pd.read_csv('mydata2.csv')

indexnames=data[data['statedt'] > 20210630].index
data.drop(indexnames, inplace=True)

d_newly_confirmed=data.iloc[:,2]
d_daily_recovered=data.iloc[:,3]
d_daily_death=data.iloc[:,4]
d_quarantined=data.iloc[:,5]

d_cum_death=data.iloc[:,6]

d_quarantined_ma=data.iloc[:,9]
d_newly_confirmed_ma=data.iloc[:,10]
d_daily_recovered_ma=data.iloc[:,11]
d_daily_death_ma=data.iloc[:,12]




data

# -

TIME_R=[]
for j in range(0,len(d_cum_death)):
    TIME_R.append(datetime.datetime(2020,2,5)+datetime.timedelta(j)) 

# +
fig, axes = plt.subplots(figsize=(20,10), nrows=2, ncols=2)
axes[0,0].plot(TIME_R, d_newly_confirmed_ma, c='r', label='data')
axes[0,0].set_title('Newly Confirmed')



axes[0,1].plot(TIME_R, d_quarantined_ma, c='r', label='data')
axes[0,1].set_title('Daily Quarantined')


axes[1,0].plot(TIME_R, d_daily_death_ma, c='r', label='data')
axes[1,0].set_title('Daily Death')

axes[1,1].plot(TIME_R, d_cum_death, c='r', label='data')
axes[1,1].set_title('Cumulative Death')

axes[0,0].grid(True)
axes[0,1].grid(True)
axes[1,0].grid(True)
axes[1,1].grid(True)

plt.savefig('KOR data')
