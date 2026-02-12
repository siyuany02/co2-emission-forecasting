from statsmodels.tsa.stattools import adfuller # 平稳性检测
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # 画acf, pacf图
from statsmodels.tsa.arima_model import ARIMA # ARIMA模型
from statsmodels.graphics.api import qqplot # 画qq图
from scipy.stats import shapiro # 正态检验
import statsmodels.tsa.stattools as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import statsmodels
import seaborn as sns
import matplotlib.pylab as plt
from scipy import  stats

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

file1 = pd.read_csv("*****.csv")
d1 = file1[['year', 'co2']]
data1 = d1.set_index('year')
n_sample = data1.shape[0]
n_train = int(0.95 * n_sample)+1
n_forecast = n_sample - n_train
ts_train = data1.iloc[:n_train]['co2']
ts_test = data1.iloc[n_train:]['co2']
plt.plot(data1)
plt.title('co2 emission ')
plt.show()

diff_df = data1.copy()
diff_df.index=data1.index
diff_df['diff_1'] =  diff_df.diff(1).dropna()
diff_df['diff_2'] = diff_df['diff_1'].diff(1).dropna()
diff_df.plot(subplots=True,figsize=(18,20),title = 'SWE difference')
plt.savefig('SWE difference')
plt.show()

from statsmodels.tsa.stattools import adfuller as ADF
print(ADF(data1))
print(ADF(diff_df['diff_1'].dropna()))
print(ADF(diff_df['diff_2'].dropna()))
ts_diff=data1.copy()
ts_diff.index=data1.index
ts_diff=data1.diff(2).dropna()

from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(ts_diff, lags = 20)

rom statsmodels.graphics.tsaplots import plot_acf, plot_pacf
pacf = plot_pacf(ts_diff, lags=40)
plt.title('PACF')
plt.savefig('SWE’s PACF')
pacf.show()
acf = plot_acf(ts_diff, lags=40)
plt.title('ACF')
plt.savefig('SWE’s ACF')
acf.show()

import itertools
import numpy as np
import seaborn as sns
p_min = 0
d_min = 0
q_min = 0
p_max = 8
d_max = 2
q_max = 8
 
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
 
for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
    if p==0 and d==0 and q==0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
 
    try:
        model = sm.tsa.ARIMA(ts_train, order=(p, d, q),
                               #enforce_stationarity=False,
                               #enforce_invertibility=False,
                              )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)
results_bic

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 )
ax.set_title(' BIC in SWE ')
plt.savefig('BIC in SWE')
plt.show()

from statsmodels.tsa.arima_model import ARIMA
model = sm.tsa.arima.ARIMA(ts_train, order=(0,2,1))
result = model.fit()
result.summary()
predict = result.predict(0,216)
predict

list1 = []
list2 = []
for i in range(0,len(predict)):
    a = predict[i]
    list1.append(a)
    list2.append(1834+i)
list1,list2

plt.figure(figsize=(12, 8))
plt.title('co2 emssion in SWE')
plt.plot(data1,label='Real')
plt.plot(list2,list1,label='predicted')
plt.legend()
plt.savefig('co2 emssion in SWE')
plt.show()

from statsmodels.tsa.arima_model import ARIMA
# ARIMA(data, order=(p, d, q))
model = sm.tsa.arima.ARIMA(ts_train, order=(6,2,3))
result = model.fit()
result.summary()

predict = result.predict(0,216)
predict

list3 = []
list4 = []
for i in range(0,len(predict)):
    a = predict[i]
    list3.append(a)
    list4.append(1834+i)
list3,list4

plt.figure(figsize=(12, 8))
plt.title('co2 emssion in SWE')
plt.plot(data1,label='Real')
plt.plot(list2,list1,label='predicted')
plt.legend()
plt.show()