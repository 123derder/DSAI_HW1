if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
print(args.training)
print(args.output)
    
#!/usr/bin/env python
# coding: utf-8

# In[104]:


import warnings
import itertools
import csv
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.style.use('fivethirtyeight')
df = pd.read_csv(args.training)


################ SAMRIMA model #####################
# In[105]:

#构建训练集测试集
data=pd.Series(df.values[:,0])
for i in range(0,len(data)):
    if data[i]>6000:
        data[i]=4500
    data[i]-=0

#y=data[:126]
y=data

#6 2 9 -300
# 6 3 9  -500
# 2 2 14
#seasonal_order=(0, 1, 1, 7),
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(6, 2, 9),
                                #enforce_stationarity=False,
                                enforce_invertibility=False)
 
results = mod.fit()



# In[160]:


pred_uc = results.get_forecast(steps=15)  # retun out-of-sample forecast 
# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

for i in range(0,15):
    pred_uc.predicted_mean[len(data)+i]-=300
    

ax = data.plot(label='observed', figsize=(30,6))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
#ax.set_xlabel('Date')
#ax.set_ylabel('CO2 Levels')
 
plt.legend()
plt.show()

y_forecasted=pred_uc.predicted_mean

print("y_forecasted")
print(y_forecasted)

date=[20220330,20220331,20200401,20200402,20200403,20200404,20200405,20200406,20200407,20200408,20200409,20200410,20200411,20200412,20200413]
with open(args.output,'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['date','operating_reserve(MW)'])
    for i in range(0,len(date)):
        writer.writerow([str(date[i]),str(int(y_forecasted[len(y)+i]))])
        print(str(y_forecasted[len(y)+i]))