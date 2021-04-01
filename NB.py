import numpy as np
import pandas as pd
import copy

df = pd.read_csv('data.csv')
X = df.iloc[:-1, 1:]
y = df.iloc[:-1, -1]

test = df.iloc[-1:,1:]
s_outlook_p=0
s_outlook_n=0
s_p = 0
s_n = 0
s_wind_p = 0
s_wind_n = 0
for i in X.index:
    if X.at[i,'play']=='yes':
        outlook = X.at[i,'outlook']
        s_p+=1
        if outlook == test.at[14,'outlook']:
            s_outlook_p+=1
        if X.at[i,'wind'] == test.at[14,'wind']:
            s_wind_p+=1

    if X.at[i, 'play'] == 'no':
        outlook = X.at[i, 'outlook']
        s_n += 1
        if outlook == test.at[14,'outlook']:
            s_outlook_n += 1
        if X.at[i, 'wind'] == test.at[14,'wind']:
            s_wind_n += 1

p_wind_p = s_wind_p/s_p
p_wind_n = s_wind_n/s_n
p_outlook_p = s_outlook_p/s_p
p_outlook_n = s_outlook_n/s_n


temp_p = []
temp_n=[]
hum_p=[]
hum_n=[]

for i in X.index:
    if X.at[i,'play']=='yes':
        temp_p.append(X.at[i,'temperature'])
        hum_p.append(X.at[i,'humidity'])
    if X.at[i,'play']=='no':
        temp_n.append(X.at[i,'temperature'])
        hum_n.append(X.at[i,'humidity'])

mean_temp_p = np.mean(temp_p)
std_temp_p = np.std(temp_p)
mean_temp_n = np.mean(temp_n)
std_temp_n = np.std(temp_n)

mean_hum_p = np.mean(hum_p)
std_hum_p = np.std(hum_p)
mean_hum_n = np.mean(hum_n)
std_hum_n = np.std(hum_n)

temp = test.at[14,'temperature']
hum = test.at[14,'humidity']
p_temp_p = np.exp((-1/2)*((temp - mean_temp_p)**2)/(std_temp_p**2))/(np.sqrt(2*np.pi)*std_temp_p)
p_temp_n = np.exp((-1/2)*((temp - mean_temp_n)**2)/(std_temp_n**2))/(np.sqrt(2*np.pi)*std_temp_n)
p_hum_p = np.exp((-1/2)*((hum - mean_hum_p)**2)/(std_hum_p**2))/(np.sqrt(2*np.pi)*std_hum_p)
p_hum_n = np.exp((-1/2)*((hum - mean_hum_n)**2)/(std_hum_n**2))/(np.sqrt(2*np.pi)*std_hum_n)


pp = p_hum_p * p_wind_p * p_outlook_p * p_temp_p
pn = p_outlook_n * p_hum_n * p_wind_n * p_temp_n

if pp>pn:
    print("True")
else:
    print("False")

# SINCE THERE ARE TWO CONTINUOUS ATTRIBUTES WE NEED TO LOOK AT THOSE AS WELL

