
# coding: utf-8

# In[ ]:


###FE-595 homework Assignment-1
##Python refresher

##Name - Monica Vijaywargi
## CWID - 10423110


##Importing the matplotlib and numpy lib as plt and np respectively
import matplotlib.pyplot as plt
import numpy as np


## the plot goes from 0 to 360 (2pi) for cycles of sine and cosine
period = np.arange(0,2*np.pi,0.01)   # start,stop,step

#Sine graph
sin = np.sin(period)

# cosine graph
cos = np.cos(period)

#plotting sin and cosine on the same axis
plt.plot(period,sin,period,cos)

##creating legends
plt.subplot().legend(['Sine','Cosine'])

# Creating X and Y axes
plt.subplot().axhline(y=0,color='k')
plt.subplot().axvline(x=0,color='k')

# Displaying the plot
plt.show()

