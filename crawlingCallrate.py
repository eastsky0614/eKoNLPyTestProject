
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import csv


data=pd.read_csv("callrate_dataset.csv", names=['date','callrate'], encoding='utf-8') #20040101 ~ 20171229
df=pd.DataFrame(data) # 휴일(Nan값)이 제거된 dataset. 3277 : 0 ~ 3276
cr=df['callrate']


#Data periods setting. 휴일이 제거되었으므로 20일을 한달 기준. diff(callrate)
cr_arr=np.array(df['callrate'])
period  = cr_arr[23:3255]       # 20050103 ~ 20171229, 3232
period23= cr_arr[0:3232]        # 20041201 ~ 20051130, 3232
period20= cr_arr[2:3234]        # 20041203 ~ 20051130, 3232

diff23 = period-period23        # numpy.array, 3232
diff20 = period-period20        # numpy.array, 3232


#Classifiacation, (NaN) setting
label23=[]                      #list
for b,a in zip(period, period23):
    if b-a > 0:
        label23.append("1")     #pos
    elif b-a < 0:
        label23.append("0")     #neg
    # elif b-a == "0" :
    #     diff2.append("dl")
    else:
        label23.append("del")

#print(label23)        #3232, dl=582, list
#print(len(label23))



label20=[]                      #list
for b,a in zip(period, period20):
    if b-a > 0:
        label20.append("1")     #pos
    elif b-a < 0:
        label20.append("0")     #neg
    # elif b-a == "0" :
    #     diff2.append("dl")
    else:
        label20.append("del")

#print(label20)        #3232, dl=613, list
#print(len(label20))




# New DataFrame
data2=pd.read_csv("callrate_dateset0517.csv", names=['date'], encoding='utf-8') #20050101 ~ 20171229
df2=pd.DataFrame(data2) #3232


df2['date'] = df['date']
df2['label23']=label23
df2['label20']=label20
df2['diff23']=diff23
df2['diff20']=diff20
df2['callrate']=cr

df2 = df2[df2.label23 != 'del'] # period-period23 = 0값 제거


data_table=pd.DataFrame(df2, columns=['date','label23','label20','diff23' ,'diff20','callrate'])
print(data_table)


# Save
# data_table.to_csv("callrate_name.csv")


# 인코딩확인 check
# corpusData = pd.read_table("callrate_name.csv", sep=',', header=None, names=None, encoding='utf-8')
# corpusData = np.array(corpusData)
# print(corpusData) # numpy.ndarray / object

