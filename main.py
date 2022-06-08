from statsmodels import api as sm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as manimation

plt.style.use('seaborn-poster')

Fs = 1  # sampling rate
Ts = 1.0/Fs # sampling interval
t = np.arange(0,100,Ts) # time vector
ff = 0.1;   # frequency of the signal
# create the signal
y = np.sin(2*np.pi*ff*t + 5)

plt.figure(figsize = (10, 8))
plt.plot(t, y)
plt.show()

# get the autocorrelation coefficient
acf = sm.tsa.acf(y, nlags=len(y))

plt.figure(figsize = (10, 8))
lag = np.arange(len(y))
plt.plot(lag, acf)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')

