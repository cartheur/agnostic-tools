## Autocorrelation in practise
As described in [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) her e is code to create an animation of how the autocorrelation method works. 


```python
from statsmodels import api as sm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as manimation

plt.style.use('seaborn-poster')
%matplotlib inline
```

## Generate a sine signal

To begin, generate a sinusoidal (sine) wave with frequency 0.1 Hz using a sampling rate of 1 Hz, then plot:


```python
Fs = 1  # The sampling rate
Ts = 1.0/Fs # The sampling interval
t = np.arange(0,100,Ts) # A time vector
ff = 0.1;   # Frequency of the signal
# Create the signal
y = np.sin(2*np.pi*ff*t + 5)
# Plot
plt.figure(figsize = (10, 8))
plt.plot(t, y)
plt.show()
```

![png](/images/fig.1.png)

## Plot the autocorrelation using statsmodels

Plot the autocorrelation using an existing package - [statsmodels](http://statsmodels.sourceforge.net/). The idea of autocorrelation is to provide a measure of similarity between a signal and itself with a time-delay relative. 


```python
# Get the autocorrelation coefficient
acf = sm.tsa.acf(y, nlags=len(y))
```


```python
plt.figure(figsize = (10, 8))
lag = np.arange(len(y))
plt.plot(lag, acf)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
```

![png](/images/fig.2.png)


## Creating the animation to illustrate the autocorrelation method

It is a practical execution of a simple method in the time domain where a signal of interest (or in noise) is understood relative to its position in the time-domain. Contrary to what one learns in physics, not all spectra are continuous. By shifting the signal to incude a time lag (gaining a new _family time_) and calculate the correlation with the original signal the function will [shift the signal in the frequency domain](http://qingkaikong.blogspot.com/2016/03/shift-signal-in-frequency-domain.html). 


```python
def nextpow2(i):
    '''
    Find the next power 2 number for FFT
    '''
    
    n = 1
    while n < i: n *= 2
    return n

def shift_signal_in_frequency_domain(datin, shift):
    '''
    This is function to shift a signal in frequency domain. 
    In the frequency domain, multiply the signal with its phase shift. 
    '''
    Nin = len(datin) 
    
    # Get the next power 2 number for fft
    N = nextpow2(Nin +np.max(np.abs(shift)))
    
    # Perform the fft
    fdatin = np.fft.fft(datin, N)
    
    # Get the phase shift for the signal, shift here is D in the above explaination
    ik = np.array([2j*np.pi*k for k in xrange(0, N)]) / N 
    fshift = np.exp(-ik*shift)
        
    # Multiply the signal with the shift and transform it back to time domain
    datout = np.real(np.fft.ifft(fshift * fdatin))
    
    # Only get the data have the same length as the input signal
    datout = datout[0:Nin]
    
    return datout
```

For the animation, shift the signal 1 at a step, and calculate the correlation normalized by the largest value - where the two signals overlap with each other, e.g., at zero delay. 


```python
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Animation', artist='Meplotlib',
                comment='Animation tools')
writer = FFMpegWriter(fps=15, metadata=metadata)
```


```python
lags = []
acfs = []
norm = np.dot(y, y)
fig = plt.figure(figsize = (10, 8))
n_frame = len(y)

def updatefig(i):
    '''
    A helper function to plot the two figures, the top panel is the time domain signal with the red signal showing the shifted signal. The bottom figure is the one corresponding to the autocorrelation from the above figure. 
    '''
    fig.clear()
    # Apply a shift to the signal
    y_shift = shift_signal_in_frequency_domain(y, i)
    plt.subplot(211)
    plt.plot(t, y, 'b')
    plt.plot(t, y_shift, 'r')
    plt.ylabel('Amplitude')
    plt.title('Lag: ' + str(i))

    plt.subplot(212)
    # Get the delay
    lags.append(i)
    # A simple autocorrelation method
    acf = np.dot(y_shift, y)
    # Add to the list with normalized value
    acfs.append(acf/norm)
    plt.plot(lags, acfs)
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.xlim(0, n_frame)
    plt.ylim(-1, 1)
    plt.draw()

# save the movie to file
anim = manimation.FuncAnimation(fig, updatefig, n_frame)
anim.save("./autocorrelation.mp4", fps=10, dpi = 300)
```

![gif](/images/autocorrelation.gif)

### Summary

By applying the method of autocorrelation, a stepwise shift of the signal by 1 and calculate the autocorrelation as: 
1. Multiply the numbers from two signals at each timestamp (You can think two signals as two arrays, and we do an elementwise multiply of the two arrays)
2. After the elementwise multiplication, we get another array, which we will sum them up to get a number - the autocorrelation. 

This is the dot product of the two signals. Yhe red signal shifted away from the very beginning of the total overlap, the two signals start to out of phase, and the autocorrelation decreasing. Due to the signal is a periodic signal, the red signal soon overlap with the blue signal again. However, the red signal shifted by certain delays, it only partially overlap with the original blue signal, therefore, the autocorrelation of the second peak is smaller than the first peak. As the red signal shifted further, the part overlap becomes smaller, which generating the decreasing trend of the peaks. 

Conversion of mp4 output to gif can be done with ```ffmpeg```:

```bash
ffmpeg -i autocorrelation_example.mp4 -vf scale=768:-1 -r 10 -f image2pipe -vcodec ppm - | convert -delay 10 -loop 0 - output.gif
```
