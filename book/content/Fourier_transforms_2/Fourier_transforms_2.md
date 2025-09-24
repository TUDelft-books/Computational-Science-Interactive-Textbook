---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Applications of the Fourier Transform

In the previous lecture notebook, we looked into detail about how the 1D FFT works in Python, and saw an example of using the FFT to detect a weak sinusoidal signal in a noisy dataset. 

In this lecture notebook, you will explore the application of the 1D FFT for filtering signals, and also learn about the 2D FFT and and application of it in calculating diffraction patterns. 

**Learning objectives:** After finishing this notebook, you should be able to:

1. Use the FFT to filter numerical data
2. Interpret artifacts in your filtered data arising from the FFT
3. Calculate the 2D Fourier transform of 2D data
4. Construct 2D images of mask patterns and calculate the far-field diffraction pattern


```python 
# Initialisation code for the notebook
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100
```


## Background information on Filtering Signals

A common way to remove noise from your data is to apply a low-pass filter: 

https://en.wikipedia.org/wiki/Low-pass_filter

This is equivalent to passing an electrical signal through the following <a href=https://en.wikipedia.org/wiki/RC_circuit>RC circuit</a>:

![image](1st_Order_Lowpass_Filter_RC.svg.png)

To predict what the output voltage is for a given input voltage, we need to know the <a href=https://en.wikipedia.org/wiki/Infinite_impulse_response>infinite impulse response</a>, which is the response of the circuit to an input an input voltage that is a delta-function $\delta(0)$. For the circuit above, the impulse response is given by (see <a href=https://en.wikipedia.org/wiki/RC_circuit#Impulse_response>here</a>):

$$
h(t) = \frac{1}{\tau} e^{-\frac{-t}{\tau}} \theta(t)
$$

where $\tau = RC$ is the RC time constant of the circuit, and $\theta(t)$ is a step function. We can see what this looks like using this code:

```python 
# Notebook code 
tau = 1
plt.figure(figsize=(12,3))
t = np.linspace(-5,5,1000)
h = np.exp(-t/tau)*(t>=0)
plt.plot(t,h)
plt.ylabel("Impulse response h(t)")
plt.xlabel("Time");
```

Once we know the impulse function of a filter, we can then calculate output $V_{out}(t)$ of the filter circuit for **any** given input voltage $V_{in}(t)$ using a convolution:

$$
V_{out}(t) = \int_{-\infty}^t V_{in}(t-\tau) h(\tau) d\tau
$$

A nice way to understand what this does is to make a plot. The output voltage $V_{out}$ at time $t_0$ is a "weighted average" of all the input voltages $V_{in}$ at times $t < t_0$ where the "weighing factor" is given by the function $h(-\tau)$ that is decaying exponentially "backwards" in time:

```python
# Notebook code 
fix, ax = plt.subplots(figsize=(12,3))
vin = np.random.normal(size=len(t))*0.05
plt.plot(t,vin,'.', color='grey', label=r"$V_{in}$")
plt.plot(t,np.exp(t)*(t<0),'r', label=r"Impulse response")
plt.fill_between(t,np.exp(t)*(t<0), alpha=0.1, color='r')
plt.xlabel("Relative Time")
ax.get_yaxis().set_visible(False)
plt.legend()
plt.title("Illustration of the convolution integral");
```

We say that this filter has a "memory" because the value of $V_{out}$ depends on the values of $V_{in}$ in the past. We also say that it is "causal" because the value of $V_{out}$ at time $t$ does not depend on values of $V_{in}$ in the future.

_(Aside: Any real filter you make in the lab will of course be "causal", but in software you can easily make a non-causal filter if you want...)_

While it is possible to apply a low pass filter by convolving $V_{in}(t)$ with $h(\tau)$ using, for example, two nested `for` loops, it is far more common in practice to perform filtering in the frequency domain.

To see how this works, we first look at an important mathematical result, known as the <a href=https://en.wikipedia.org/wiki/Convolution_theorem>convolution theorem</a>. Consider now a completely general function $c(x)$ given by the convolution of two function $f(x)$ and $g(x)$: 

$$
c(x) = \int_{-\infty}^{\infty} f(y) g(x-y) dy
$$

If $\tilde f(\omega)$ and $\tilde g(\omega)$ are the Fourier transforms of $f(x)$ and $g(x)$ respectively, then the convolution theorem states that the Fourier transform $\tilde c(\omega)$ of $c(x)$ is given by: 

$$
\tilde c(\omega) = \tilde f(\omega) \ \tilde g(\omega)
$$

This is a big deal! Why? In the first equation above, for each point in $h(x)$ I need to calculate a full integral. If these are all vectors of size $N$, this integral takes $N$ steps, and so calculating $h(x)$ takes order $N^2$ steps. In the second equation, this is only **point-wise multiplication** of the two functions! So if I work in Fourier space, calculating the Fourier transform $\tilde c(\omega)$ takes only order $N$ steps. If $N$ is $10^6$, this is the difference between $10^6$ steps and $10^{12}$ steps, **which is a big difference!** If a step takes 1 ms, this is the difference between a calculation of 3.6 hours or 31 years!

(Of course, we still need to take the Fourier transform and inverse Fourier transform, but thanks to Gauss' genius, this scales only like $N\log N$, so only $6\times10^6$ steps.)

To calculate $V_{out}$ of our filter, we then need to find the Fourier transform of our impulse response function $h(t)$. For the low pass filter above, this is given by:

$$
\tilde h(\omega) = \frac{1}{1+i\omega \tau}
$$

The Fourier transform $\tilde h(\omega)$ of the impulse response function is also referred to as the <a href=https://en.wikipedia.org/wiki/Transfer_function>Transfer Function</a>. In our case, it tells you how the amplitude and phase of an oscillating input signal is related to the amplitude and phase of the output signal. Using <a href=https://en.wikipedia.org/wiki/Phasor>phasor notation</a> $V(t) = \tilde V e^{-i\omega t}$, the input and output signals are related by:

$$
\tilde V_{out}(\omega) = \tilde h(\omega) \tilde V_{in}(\omega)
$$

You can see a plot of the transfer function for an RC filter here: 

```python
# Notebook code 
tau = 1
w = np.geomspace(1e-3,1e3, 100)
h = 1/(1+1j*w*tau)

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(w,np.abs(h))
plt.xlabel("$\omega$")
plt.ylabel("$|h(\omega)|$")
plt.xscale('log')
plt.yscale('log')
plt.grid()

plt.subplot(122)
plt.plot(w, np.angle(h)/np.pi*180)
plt.xscale('log')
plt.xlabel("$\omega$")
plt.ylabel(r"Phase angle of $\tilde h$ (degrees)")
plt.grid()
plt.yticks(np.arange(-90,1,15));
```

To get back to the output singal $V_{out}(t)$, all we have to do is then apply the inverse Fourier transform:

$$
V_{out}(t) = \int_{-\infty}^{\infty} \tilde V_{in}(\omega) \tilde h(\omega) e^{i\omega t} d\omega
$$

In python, you would first use `np.fft.fft()` to calculate the FFT of the input signal, multiply this by the filter transfer function you want to use, and then take the inverse FFT using `np.fft.ifft()`. Done!

## Filtering Exercises

You will start by calculating the result of filtering a square pulse using a low-pass filter. 

**Exercise 1(a):** Make a square voltage pulse:  

$$
V(t) = 
\begin{cases}
0 &  \text{for } 0 <t <=50 \text{ s}\\
1 &  \text{for } 50 <t <=100 \text{ s} \\
0 &  \text{for } 100 <t <=150 \text{ s}
\end{cases}
$$

Your pulse should include 1000 points

```python
# t = np.linspace(...)

# If you are well versed with vectorized comparison operators, you
# can create v in one line! Otherwise, a for loop is OK too...
# v = ....

plt.plot(t,v)
plt.ylabel("Voltage (V)")
plt.xlabel("Time (s)")

answer_8_1a_1 = np.copy(t)
answer_8_1a_2 = np.copy(v)
```

```python
question = "answer_8_1a"
num = 2

to_check = [question + "_%d" % (n+1) for n in range(num)]
feedback = ""
passed = True
for var in to_check:
    res, msg = check_answer(eval(var), var)
    passed = passed and res
    print(msg); feedback += msg + "n"

assert passed == True, feedback
```

**Solution:**
``` python tags=["hide-input"] 
# t = np.linspace(...)
### BEGIN SOLUTION
t = np.linspace(0,150,1000)
### END SOLUTION

# If you are well versed with vectorized comparison operators, you
# can create v in one line! Otherwise, a for loop is OK too...
# v = ....
### BEGIN SOLUTION
v = (t > 50) * (t <= 100)
### END SOLUTION

plt.plot(t,v)
plt.ylabel("Voltage (V)")
plt.xlabel("Time (s)")

answer_8_1a_1 = np.copy(t)
answer_8_1a_2 = np.copy(v)
```


**Exercise 1(b):** Write a function to use a FFT to apply a low-pass filter to the data. You filter should have a time constant $\tau = 5$ seconds. The cell includes code to make a plot of the original data along with the filtered data. 

_Note:_ When using `np.fft.ifft()`, it will return a complex-valued array even if the data has no imaginary component (as it should if you have done your filtering correctly). For plotting, you will need to use `np.real()` to convert your array to real values before plotting (if you forget, matplotlib will give you a warning messgae). You should therefore have your `low_pass` function return only the real-valued part of the filtered data. 

```python
def low_pass(v, tau, dt):
    # ...

vfilt = low_pass(v, 5, t[1]-t[0])

plt.plot(t,v, label="$V_{in}$")
plt.ylabel("Voltage (V)")
plt.xlabel("Time (s)")
plt.plot(t, vfilt, label="$V_{out}$");
plt.legend()

answer_8_1b_1 = np.copy(vfilt)
```

```python
question = "answer_8_1b"
num = 1

to_check = [question + "_%d" % (n+1) for n in range(num)]
feedback = ""
passed = True
for var in to_check:
    res, msg = check_answer(eval(var), var)
    passed = passed and res
    print(msg); feedback += msg + "n"

assert passed == True, feedback
```

**Solution:**
``` python tags=["hide-input"] 
def low_pass(v, tau, dt):
    ### BEGIN SOLUTION
    v_t = np.fft.fft(v)
    f = np.fft.fftfreq(len(v), d=dt)
    h = 1/(1+1j*2*np.pi*f*tau)
    vfilt_t = v_t*h
    # When we invert the FFT, we should have a real number
    # But since the np.fft.fft() function is designed to 
    # also work with complex valued data, we need to take 
    # the real part. 
    vfilt = np.real(np.fft.ifft(vfilt_t))
    return vfilt
    ### END SOLUTION

vfilt = low_pass(v, 5, t[1]-t[0])

plt.plot(t,v, label="$V_{in}$")
plt.ylabel("Voltage (V)")
plt.xlabel("Time (s)")
plt.plot(t, vfilt, label="$V_{out}$");
plt.legend()

answer_8_1b_1 = np.copy(vfilt)
```


**Exercise 1(c):** Use your function to apply a low pass filter with $\tau = 20$ seconds. As usual, your plot should have appropriate axis labels, and a legend.

```python
# vfilt = low_pass(....)
# plt.plot(....)
# .....

answer_8_1c_1 = np.copy(vfilt)
```

```python
question = "answer_8_1c"
num = 1

to_check = [question + "_%d" % (n+1) for n in range(num)]
feedback = ""
passed = True
for var in to_check:
    res, msg = check_answer(eval(var), var)
    passed = passed and res
    print(msg); feedback += msg + "n"

assert passed == True, feedback
```

**Solution:**
``` python tags=["hide-input"] 
# vfilt = low_pass(....)
# plt.plot(....)
# .....
### BEGIN SOLUTION
vfilt = low_pass(v, 20, t[1]-t[0])
plt.plot(t,v, label="$V_{in}$")
plt.ylabel("Voltage (V)")
plt.xlabel("Time (s)")
plt.plot(t, vfilt, label="$V_{out}$");
plt.legend()
### END SOLUTION

answer_8_1c_1 = np.copy(vfilt)
```


What is strange about your filtered function at $t=0$? What is going on? 

**Exercise 1(d):** We can solve the problem we saw in Exercise 1(c) by using zero-padding of the array during the Fourier transform. Read the documentation page of the numpy fft to learn how to do this:

https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html

Write a new version of the low pass function `low_pass2()` function, this time padding this signal you are filtering with as many points as it already contains. If you do this correctly, it will eliminate the "boundary" effects from your FFT we saw above.  The ploting code to make a plot of your low-pass filtered data after filtering with the new function.

```python
def low_pass2(v, tau, dt):
    # ...

vfilt2 = low_pass2(v, 20, t[1]-t[0])
plt.plot(t,v, label="$V_{in}$")
plt.ylabel("Voltage (V)")
plt.xlabel("Time (s)")
plt.plot(t, vfilt2, label="$V_{out}$");
plt.legend()

answer_8_1d_1 = np.copy(vfilt2)
```

```python
question = "answer_8_1d"
num = 1

to_check = [question + "_%d" % (n+1) for n in range(num)]
feedback = ""
passed = True
for var in to_check:
    res, msg = check_answer(eval(var), var)
    passed = passed and res
    print(msg); feedback += msg + "n"

assert passed == True, feedback
```

**Solution:**
``` python tags=["hide-input"] 
def low_pass2(v, tau, dt):
    ### BEGIN SOLUTION
    n = 2*len(v)
    v_t = np.fft.fft(v, n=n)
    f = np.fft.fftfreq(n, d=dt)
    h = 1/(1+1j*2*np.pi*f*tau)
    vfilt_t = v_t*h
    vfilt = np.real(np.fft.ifft(vfilt_t, n=n))
    return vfilt[0:len(v)]
    ### END SOLUTION

vfilt2 = low_pass2(v, 20, t[1]-t[0])
plt.plot(t,v, label="$V_{in}$")
plt.ylabel("Voltage (V)")
plt.xlabel("Time (s)")
plt.plot(t, vfilt2, label="$V_{out}$");
plt.legend()

answer_8_1d_1 = np.copy(vfilt2)
```


## 2D Fourier transforms 

So far, we have been considering the Fourier transform of one-dimensional data, and thinking in particular about the case of signals varying in time, for which the Fourier transform gives a decomposition of the signal in the frequency domain. 

In 2D, it is more typical to take the Fourier transform of spatial data (for example, of an image in space). In this case, the units of the axis of your Fourier transform is not frequency, but instead a two-dimensional <a href=https://en.wikipedia.org/wiki/Wave_vector>wave vector</a>, usually denoted by letter $k$, with units of inverse distance m$^{-1}$. For a 2D FT, the wave vector is a two-dimensional vector $\bf{k}$ with vector components $k_x$ and $k_y$.

The 2D FT is used extensively in solid-state physics, and also in image analysis. 

Here, we will explore two applications of the 2D Fourier transform for the calculation of Fraunhofer diffraction patterns.

### Fraunhofer Diffraction

Fraunhofer diffraction considers the very-far field scattering of a colimated light off of a planar object that is much smaller than the distance to the screen (and smaller than the diameter of the beam). A classical example of 1D Fraunhofer diffraction is the single and double slit diffraction patterns. Here, we will explore the Fraunhofer diffraction patterns of 2D planar masks. 

Calculating the Fraunhofer diffraction pattern is actually pretty simple: in the far field limit, the image of the light's intensity on the screen is directly given by the square magnitude of the Fourier transform of the shape of the aperture.

Note that as long as the assumptions above hold, the image of the diffraction pattern on the screen is independent of the wavelength: changing from a green to red laser pointer, for example, changes only the size / scaling of the image (the dots on the wall will be further apart for a green laser than a red one). 

**Exercise 2(a):** We will aim to create a "mask" for calculating a diffraction pattern. Your mask should consist of an array of $(N+1)\times(N+1) = 2001\times2001$ points, corresponding to $N=2000$.

To do this, we will use the `meshgrid()` command to create 2D arrays filled with X and Y coordinates. Have your X and Y coordinates run from -1000 to 1000 nanometers.

To understand exactly what `meshgrid()` does, the following code will make a plot of the two matrices size by side using `subplot(121)` and `subplot(122)` together with `imshow()`. 

```python
N = 2000

# Lower case letters correspond to the 1D array of coordinates in the two directions
x = np.linspace(-N/2,N/2,N+1)
y = np.linspace(-N/2,N/2,N+1)

# Upper case letters correspont to the 2D matrices containing the X or Y coordinates
X, Y = np.meshgrid(x, y)

# This we need to tell python what the range of the image plot are
extent = [x[0],x[-1], y[0], y[-1]]

# We will use the subplots command to make two plots beside each other
plt.subplots(figsize=(10,3.5))
plt.subplot(121)
plt.imshow(X,origin='lower',extent=extent)
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.subplot(122)
plt.imshow(Y,origin='lower', extent=extent)
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.tight_layout()
```

What do you see in the plots? What do your `X` and `Y` matrices look like?



**Exercise 2(b):** Use your X and Y matrices to make a mask matrix M in which all points should be 1, except for points with a radius of $r > 50$ points from the center of the array, which should be zero. The code will make two plots of your mask: one of the full matrix, and one zoomed into the range of a 200x200 pixel zoom at the center of the image. The mask value of 1 will indicate where the light will be transmitted, and a mask value of 0 will indicate where the light will be blocked, and so this corresponds to the diffraction through a circular aperture. 

```python
# It seems like this is magic, but using the right matrices, it does just work!
# M =  _____ < 50

plt.subplots(figsize=(10,3.5))

# The full plot
plt.subplot(121)
exts = [x[0], x[-1], y[-1], y[0]]
plt.imshow(M, extent=exts)
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")

# The zoom
plt.subplot(122)
zoom = 200
start = int(len(x)/2-zoom/2)
end = start + zoom
M_zoom = M[start:end,start:end]
exts_zoom = [x[start], x[end], y[start], y[end]]
plt.imshow(M_zoom, extent=exts_zoom)
plt.xlabel("x")
plt.ylabel("y")

answer_8_2b_1 = np.copy(M)
```

```python
question = "answer_8_2b"
num = 1

to_check = [question + "_%d" % (n+1) for n in range(num)]
feedback = ""
passed = True
for var in to_check:
    res, msg = check_answer(eval(var), var)
    passed = passed and res
    print(msg); feedback += msg + "n"

assert passed == True, feedback
```

**Solution:**
``` python tags=["hide-input"] 
# It seems like this is magic, but using the right matrices, it does just work!
# M =  _____ < 50
### BEGIN SOLUTION
M =  np.sqrt(X**2 + Y**2) < 50
### END SOLUTION

plt.subplots(figsize=(10,3.5))

# The full plot
plt.subplot(121)
exts = [x[0], x[-1], y[-1], y[0]]
plt.imshow(M, extent=exts)
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")

# The zoom
plt.subplot(122)
zoom = 200
start = int(len(x)/2-zoom/2)
end = start + zoom
M_zoom = M[start:end,start:end]
exts_zoom = [x[start], x[end], y[start], y[end]]
plt.imshow(M_zoom, extent=exts_zoom)
plt.xlabel("x")
plt.ylabel("y")

answer_8_2b_1 = np.copy(M)
```


**Exercise 2(c):** Now calculate and plot the diffraction pattern from your mask. Plot a zoom in of the diffraction pattern around the center 200 pixels of the image. Your axis labels on your diffraction pattern should correspond to wavenumbers with the appropriate units. (Hint: make sure to use fftshift, which also works on 2D arrays!)

```python
# First, we take the FFT of the mask
Mt = np.fft.fft2(M)

# Now translate this into a diffraction pattern
# diffraction_pattern = ....

# Calculate the zoom 
zoom=100
# start=....
# end=....

# Extents: this one we have to think about carefully! 
# A good trick: what is the largest wave number k_max? 
# Your full image will then run from -k_max to k_max. From this,
# you can work out the extents 

# k_max = ....
# k_zoom = k_max * zoom / len(x)
# extent_zoom = [ , , , ]

# Now the 
plt.imshow(diffraction_pattern, extent=exts)
# plt.xlabel(...)
# plt.ylabel(...)
plt.colorbar()

answer_8_2c_1 = np.copy(diffraction_pattern)
answer_8_2c_2 = k_max
answer_8_2c_3 = k_zoom
answer_8_2c_4 = np.copy(extent_zoom)
```

```python
question = "answer_8_2c"
num = 4

to_check = [question + "_%d" % (n+1) for n in range(num)]
feedback = ""
passed = True
for var in to_check:
    res, msg = check_answer(eval(var), var)
    passed = passed and res
    print(msg); feedback += msg + "n"

assert passed == True, feedback
```

**Solution:**
``` python tags=["hide-input"] 
# First, we take the FFT of the mask
Mt = np.fft.fft2(M)

# Now translate this into a diffraction pattern
# diffraction_pattern = ....
### BEGIN SOLUTION
diffraction_pattern = np.abs(np.fft.fftshift(Mt))**2
### END SOLUTION

# Calculate the zoom 
zoom=100
# start=....
# end=....
### BEGIN SOLUTION
start = int(len(x)/2-zoom/2)
end = start+zoom
### END SOLUTION

# Extents: this one we have to think about carefully! 
# A good trick: what is the largest wave number k_max? 
# Your full image will then run from -k_max to k_max. From this,
# you can work out the extents 

# k_max = ....
# k_zoom = k_max * zoom / len(x)
# extent_zoom = [ , , , ]
### BEGIN SOLUTION
k_max = 1/(x[1]-x[0])/2 # the nyquist frequency
k_zoom = k_max * zoom / len(x)
extent_zoom = [ -k_zoom , k_zoom, -k_zoom, k_zoom]
### END SOLUTION

# Now the 
plt.imshow(diffraction_pattern, extent=exts)
# plt.xlabel(...)
# plt.ylabel(...)
### BEGIN SOLUTION
plt.xlabel("k$_x$ (nm$^{-1}$)")
plt.ylabel("k$_y$ (nm$^{-1})$")
### END SOLUTION
plt.colorbar()

answer_8_2c_1 = np.copy(diffraction_pattern)
answer_8_2c_2 = k_max
answer_8_2c_3 = k_zoom
answer_8_2c_4 = np.copy(extent_zoom)
```


**Exercise 2(d)** Here, you will use the ipywidgets library to make an interactive zoom of your diffraction pattern. Fill in the code in the update function to generate the zoom as specified in the number of pixels around the center of the image. 

After running the code, you will get an image with two sliders that allow you to adjust the zoom and the maximum of the colormap (in logarithmic intervals...). The update is not super-fast: it is best to select the slider and use the arrow keys to adjust them (using the mouse will sometimes result in the image lagging behind). 

It is quite cool to zoom out and take the vmax to the lowest value in the array, particularly if you make your figure very big by using `plt.figure(figsize=(10,10))` (or some size that fills your screen nicely). 

```python
from ipywidgets import interact

# To really be able to tweak the colormap over a large range,
# we will make an array of maximum colors values that is 
# gemoetrically spaced
N = 20
vmax_array = np.geomspace(1e1,6e7,N)

def update(zoom=1000, v_index=N-1):
    vmax = vmax_array[v_index]
    # ...
    
interact(update, zoom=(20,1000,10), v_index=(1,N-1,1))
```

**Solution:**
``` python tags=["hide-input"] 
from ipywidgets import interact

# To really be able to tweak the colormap over a large range,
# we will make an array of maximum colors values that is 
# gemoetrically spaced
N = 20
vmax_array = np.geomspace(1e1,6e7,N)

def update(zoom=1000, v_index=N-1):
    vmax = vmax_array[v_index]
    ### BEGIN SOLUTION
    start = int(len(x)/2-zoom/2)
    end = start+zoom
    k_zoom = k_max * zoom / len(x)
    exts = [ -k_zoom , k_zoom, -k_zoom, k_zoom]
    plt.subplots(figsize=(10,10))
    plt.imshow(diffraction_pattern[start:end,start:end], extent=exts, vmax=vmax)
    plt.xlabel("k$_x$ (nm$^{-1}$)")
    plt.ylabel("k$_y$ (nm$^{-1})$")
    plt.colorbar()
    ### END SOLUTION
    
interact(update, zoom=(20,1000,10), v_index=(1,N-1,1))
```

Can you understand where the funky patterns you see come from?

```python tags=["auto-execute-page", "thebe-init", "hide-input"]
## Pre-loading the solutions

import sys
await micropip.install("numpy")
from validate_answers import *

with open(location):
    pass # Initially this notebook does not recognise the file unless someone tries to read it first
```
