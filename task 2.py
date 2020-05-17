#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
data = pd.read_excel(r'C:\Users\user\Downloads\spectrumcet\DS_ML_Task1\New Folder\student-math.xlsx')
data.head(400)


# In[17]:


import pandas as pd
data = pd.read_excel(r'C:\Users\user\Downloads\spectrumcet\DS_ML_Task1\New Folder\student-math.xlsx')
data.tail(n=400)
a = data[['G1','G2','G3']].tail(400)
print('all rows of last 3 colyumns:',a)
b = a.sum(axis=1)
print (b)
loc = 33
column = ('final grade')
value = b
data.head(400)
data.insert(loc,column,value,allow_duplicates = False)
data.head(400)


# In[19]:


import pandas as pd
data = pd.read_excel(r'C:\Users\user\Downloads\spectrumcet\DS_ML_Task1\New Folder\student-math.xlsx')
data.drop(['G1','G2','G3'],axis = 1, inplace = True)
data.head(n=400)


# In[23]:


import pandas as pd
from pandas import DataFrame
data = pd.read_excel(r'C:\Users\user\Downloads\spectrumcet\DS_ML_Task1\New Folder\student-math.xlsx')
a= DataFrame(data,columns=['schoolsup'])
print(a)
data.loc[data['schoolsup']== 'yes','schoolsup'] = 1
data.loc[data['schoolsup']== 'no','schoolsup'] = 0
a = data[['schoolsup']].tail(400)
print(a)

b = DataFrame(data,columns=['famsup'])
print(b)
data.loc[data['famsup']=='yes','famsup'] = 1
data.loc[data['famsup']=='no','famsup'] = 0
b = data[['famsup']].tail(400)
print(b)

c = DataFrame(data,columns=['paid'])
print(c)
data.loc[data['paid']=='yes','paid'] = 1
data.loc[data['paid']=='no' ,'paid'] = 0
c = data[['paid']].tail(400)
print(c)

d = DataFrame(data,columns=['activities'])
print(d)
data.loc[data['activities']== 'yes','activites'] = 1
data.loc[data['activities']== 'no','activities'] = 0
c = data[['activities']].tail(400)
print(d)

e = DataFrame(data,columns = ['nursery'])
print(e)
data.loc[data['nursery']== 'yes','nursery'] = 1
data.loc[data['nursery']== 'no','nursery'] = 0
e = data[['nursery']].tail(400)
print(e)

f = DataFrame(data,columns =['higher'])
print(f)
data.loc[data['higher']=='yes','higher'] = 1
data.loc[data['higher']=='no','higher'] = 0
f = data[['higher']].tail(400)
print(f)

g = DataFrame(data,columns =['internet'])
print(g)
data.loc[data['internet']=='yes','internet'] = 1
data.loc[data['internet']=='no','internet'] = 0
g = data[['internet']].tail(400)
print(g)

h = DataFrame(data,columns = ['romantic'])
print(h)
data.loc[data['schoolsup'] =='yes','romantic'] = 1
data.loc[data['romantic']=='no','romantic'] = 0
h = data[['romantic']].tail(400)
print(h)




# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_excel(r'C:\Users\user\Downloads\spectrumcet\DS_ML_Task1\New Folder\student-math.xlsx')
data.tail(n=400)
a = data[['G1','G2','G3']].tail(400)
print('all rows of last 3 colyumns:',a)
b = a.sum(axis=1)
print (b)
loc = 33
column = ('final grade')
value = b
data.head(400)
data.insert(loc,column,value,allow_duplicates = False)
data.head(400)
d = data.plot(kind = 'scatter',y ='studytime',x = 'final grade', c='black', colormap ='viridis')
e = data.plot(kind = 'density',y ='studytime',x = 'final grade',c ='red')
f = data.plot(y ='studytime',x ='final grade',c = 'blue')


# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_excel(r'C:\Users\user\Downloads\spectrumcet\DS_ML_Task1\New Folder\student-math.xlsx')
data.tail(n=400)
a = data[['G1','G2','G3']].tail(400)
print('all rows of last 3 colyumns:',a)
b = a.sum(axis=1)
print (b)
loc = 33
column = ('final grade')
value = b
data.head(400)
data.insert(loc,column,value,allow_duplicates = False)
data.head(400)
d = data.plot(kind ='scatter',y ='studytime',x ='final grade',c='black',colormap='viridis')
d = data.boxplot(by ='final grade' , column =['studytime'],grid= False)


# In[ ]:




