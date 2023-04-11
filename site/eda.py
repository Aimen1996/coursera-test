import pandas as pd
import numpy as np
import lmdb
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

min_time=34992
max_time=39052

db_path = '/home/aimen/PycharmProjects/HydraulicSupport_pressure/'
env = lmdb.open(db_path+ "support_id_lmdb40")
txn = env.begin(write=False)
temp = []
for sample_time in range(min_time, max_time):
    data_value = txn.get(str(sample_time).encode())
    temp.append(np.float32(float(data_value)))
data=np.array(temp)

def Tstationarity(series, str):

    # series['rolling_mean'] = series[str].rolling(window=12).mean()
    # series['rolling_std'] = series[str].rolling(window=12).std()
    print(series.head(13))

    #performing adf test
    result = adfuller(series[str], autolag='AIC')
    print('result',result)
    print(f'ADF Statistic: {result[0]}')
    print(f'n_lags: {result[2]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')
    if result[0] < result[4]["5%"]:
        print("Reject Ho - Time Series is Stationary")
    else:
        print("Failed to Reject Ho - Time Series is Non-Stationary")

    #ploting
    plt.figure(figsize=(10, 6))
    plt.plot(series[str], 'r')
    # plt.plot(series['rolling_mean'], 'b')
    # plt.plot(series['rolling_std'], 'g')
    plt.xlabel('Time [s]')
    plt.ylabel('pressure')
    plt.show()

# #checking on raw data
pd_data = pd.DataFrame(data, columns = ['pressure'])
# Tstationarity(pd_data,'pressure')
# exit()

#checking on z-score
pd_data['mean']=pd_data['pressure'].mean()
pd_data['std']=pd_data['pressure'].std()
pd_data['z_score']=(pd_data['pressure']-pd_data['mean'])/pd_data['std']
# print(pd_data.head)
Tstationarity(pd_data,'z_score')
exit()


#checking differnce
pd_dif=pd_data[['pressure']]
pd_dif['shift']=pd_dif['pressure'].shift()
pd_dif['diffshift']=pd_dif['pressure']-pd_dif['shift']
# Tstationarity(pd_dif.dropna(),'diffshift')
# exit()


#checking with log
pd_log=pd_data[['pressure']]
pd_log['log']=np.log(pd_log['pressure'])
# Tstationarity(pd_log,'log')
# exit()


#checking with square_root
pd_sq=pd_data[['pressure']]
pd_sq['square']=np.sqrt(pd_sq['pressure'])
# Tstationarity(pd_sq,'square')
# exit()


# checking with cube_root
pd_cbrt=pd_data[['pressure']]
pd_cbrt['cube']=np.cbrt(pd_cbrt['pressure'])
# Tstationarity(pd_cbrt,'cube')
# exit()


#checking with log+squareroot
pd_log2=pd_log[['pressure','log']]
pd_log2['log_sqrt']=np.sqrt(pd_log2['log'])
# print(pd_log2.head())
# Tstationarity(pd_log2,'log_sqrt')
# exit()


#checking (log_sqt)-(log-sqt).shift
pd_log2['logsqrtdiff']=pd_log2['log_sqrt']-pd_log2['log_sqrt'].shift()
print(pd_log2.head())
# Tstationarity(pd_log2.dropna(),'logsqrtdiff')

#checking using z-score
# data=standarizeData(temp)

