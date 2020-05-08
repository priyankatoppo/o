#!/usr/bin/env python
# coding: utf-8

# In[2]:


from array import*
arrayScore = array('i',[100,108,112,115,150,178,143,132,190,235,253,298,328,390,257,288,393,425,458,450,473,333,452,490,495,488,543,532,590,605])
for x in arrayScore:
    print(x)


# In[5]:


import numpy as np
days = np.arange(1,31)
print(days)


# In[17]:


import matplotlib.pyplot as plt
import numpy as np
y=array('i',[100,108,112,115,150,178,143,132,190,235,253,298,328,390,257,288,393,425,458,450,473,333,452,490,495,488,543,532,590,605])
x=np.arange(1,31)
plt.xlim(0,30)
plt.ylim(100,1000)
plt.xlabel('score')
plt.ylabel('days')
plt.title('dictionary game')
plt.plot(x,y)
plt.show()


# In[13]:


import numpy as np
arrayScore = array('i',[100,108,112,115,150,178,143,132,190,235,253,298,328,390,257,288,393,425,458,450,473,333,452,490,495,488,543,532,590,605])
print("Score :",arrayScore)
print("mean of Score :",np.mean(arrayScore))


# In[15]:


import numpy as np
arrayScore = array('i',[100,108,112,115,150,178,143,132,190,235,253,298,328,390,257,288,393,425,458,450,473,333,452,490,495,488,543,532,590,605])
print("Score :",arrayScore)
print("median of Score :",np.median(arrayScore))


# In[16]:


import numpy as np
arrayScore = array('i',[100,108,112,115,150,178,143,132,190,235,253,298,328,390,257,288,393,425,458,450,473,333,452,490,495,488,543,532,590,605])
maxElement = np.amax(arrayScore)
minElement = np.amin(arrayScore)
print('max element :', maxElement)
print('min element :', minElement)


# In[ ]:




