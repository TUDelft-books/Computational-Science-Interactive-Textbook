---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Ordinary Differential Equations Part 2

In this lecture, we will explore applying the techniques for solving initial value problems that we learned in lecture 11. We will also learn how to use numerical integration of ODEs to solve boundary value problems.

**Learning objectives:** After completing this lecture, you should be able to:

1. Use numerical integration to find the steady-state of the driven damped Harmonic oscillator
2. Solve boundary-value problems using the "shooting method"
3. Use numerical integration to solve problems with nonlinear damping


```python nbgrader={"grade": false, "grade_id": "cell-58b9a41c0d368e48", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Notebook code
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

plt.rcParams['figure.dpi'] = 100
```

## The driven damped Harmonic oscillator

The first physical problem we will be interested in simulating is the driven, damped Harmonic oscillator. 

The physics of a harmonic oscillator is described by the motion of a mass free to move in one dimension that is attached to a rigid wall by a spring that exerts a restoring force:

$$ 
F_r = -kx
$$

In general, a mass will also experience a friction drag force proportional to and opposing its velocity:

$$
F_f = - c \frac{dx}{dt}
$$

where $c$ is the friction coefficient. Including a time dependent external force $F_{e}(t)$ applied to the mass, the total force becomes:

$$
F_{tot} = - k x - c \frac{dx}{dt} + F_0 \cos \omega t
$$

Using $F = ma$, we find the following second order differential equation for x:

$$
m \frac{d^2x}{dt^2} + c\frac{dx}{dt} + k x  = F_0 \cos(\omega t)
$$

For this equation, there is a <a href=https://en.wikipedia.org/wiki/Harmonic_oscillator#Steady-state_solution>steady-state solution</a> for $x(t)$ given by:

$$
x(t) = A(\omega) \cos \left[ \omega t + \theta(\omega) \right] 
$$

where $A(\omega)$ and $\theta(\omega)$ are an amplitude and phase that depend on the driving frequency, and are given by:

$$
A(\omega) & = & \frac{F_0 / m}{\sqrt{\gamma^2 \omega^2 + (\omega^2 - \omega_0^2)^2}} \\
\theta(\omega) &=& \arctan \left( \frac{\omega \gamma}{\omega^2 - \omega_0^2} \right) - \pi 
$$

where $\omega_0 = \sqrt{k/m}$ is the natural frequency of the harmonic oscillator and $\gamma = c/m$ is the damping rate.  

In this problem, you will perform numerical integration of the driven damped Harmonic oscillator (DDHO) to find its steady-state response, and compare what you find to the analytical formulas above. 

We will consider the case of $m = 1$ kg, $k = 1$ N/m, $c = 0.1$ Ns/m, and $F_0 = 1$ N. 

**Exercise 1(a)** Consider the case in which the mass is at rest and is at position $x=0$ at $t=0$, and a driving force at frequency $w = w_0$. Use the `solve_ivp()` routine to calculate $x(t)$ for $t$ from 0 to 200 seconds with 1000 points. Make a plot of your solution and discuss if it behaves in the way that you would expect. 

```python
m = 1
k = 1
c = 0.1
F0 = 1
w = 1

# I will chose y[0] = x and y[1] = v 
def dydt(t,y):
    # ...

T = 200
N = 1000
t = np.linspace(0,T,N)

x0 = 0
v0 = 0

# sol = solve_ivp(....)
# t = ...
# x = ...

# Now a plot

answer_12_1a_1 = np.copy(x)
```

```python
question = "answer_12_1a"
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
m = 1
k = 1
c = 0.1
F0 = 1
w = 1

# I will chose y[0] = x and y[1] = v 
def dydt(t,y):
    ### BEGIN SOLUTION
    x = y[0]
    v = y[1]
    return [v, -k*x/m - c*v/m + F0*np.cos(w*t)/m]
    ### END SOLUTION

T = 200
N = 1000
t = np.linspace(0,T,N)

x0 = 0
v0 = 0

# sol = solve_ivp(....)
# t = ...
# x = ...
### BEGIN SOLUTION
sol = solve_ivp(dydt, (0,T), (x0,v0), t_eval=t)
t = sol.t
x = sol.y[0,:]
### END SOLUTION

# Now a plot
### BEGIN SOLUTION
plt.plot(t,x)
plt.xlabel("t (s)")
plt.ylabel("x (m)")
### END SOLUTION

answer_12_1a_1 = np.copy(x)
```


If you've done things correctly, you should see an oscillating signal that grows in amplitude and then exponentially approaches a fixed amplitude for long times (this is referred to as the "ring-up" of the oscillator). At steady state, it looks like the signal is oscillating as we expect. Qualitatively, it looks good, but does it quantitatively agree with the analytical formulas from above for the steady state behaviour? 

If I want to check this, I need to extract the steady state response from the time trace above. How do I know after which time my simulated trace has reached steady-state? Technically, it only approaches steady state exponentially, so it will actually never completely reach steady-state in any case. 

For estimating the steady state amplitude, we could do this by calculating the height of the last maximum of the oscillations. 

**Exercise 1(b):** Find the steady state amplitude of the calculated time trace by finding the value of the amplitude of the last peak in the trace. For this, you can use the `find_peaks()` routine of `scipy`:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

to find all of the peaks of the oscillations. Make a plot of the peak values vs. time, and then extract an estimate of the steady state amplitude using the value of the last peak. 

Does it agree with the prediction of the analytical formulas above? To check this, create a function `amp(w, m, c, k, F0)` (with `w` as $\omega$) to calculate the amplitude from the formula above. Use this to calculate the theoretically predicted amplitude based on the parameters you have used in the simulation.


```python
def amp(w, m, c, k, F0):
    # ...

# peak_indices, _ = ....
# t_p = ...
# x_p = ...


# A plot

# Extract the calculated amplitude
# amp_calc = ....


# Calculate the theory prediction
# amp_theory = ....

print("Calculated steady state amplitude: ", amp_calc)
print("Predicted steady state amplitude : ", amp_theory)

answer_12_1b_1 = amp_calc
answer_12_1b_2 = amp_theory
```

```python
question = "answer_12_1b"
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
def amp(w, m, c, k, F0):
    ### BEGIN SOLUTION
    gam = c/m
    w0 = np.sqrt(k/m)
    return F0/m/np.sqrt(gam**2*w**2 + (w**2 - w0**2)**2)
    ### END SOLUTION

# peak_indices, _ = ....
# t_p = ...
# x_p = ...
### BEGIN SOLUTION
peak_indices, _ = find_peaks(x)
t_p = t[peak_indices]
x_p = x[peak_indices]
### END SOLUTION


# A plot
### BEGIN SOLUTION
plt.plot(t_p, x_p)
### END SOLUTION

# Extract the calculated amplitude
# amp_calc = ....
### BEGIN SOLUTION
amp_calc = x_p[-1]
### END SOLUTION


# Calculate the theory prediction
# amp_theory = ....
### BEGIN SOLUTION
amp_theory = amp(1, m, c, k, F0)
### END SOLUTION

print("Calculated steady state amplitude: ", amp_calc)
print("Predicted steady state amplitude : ", amp_theory)

answer_12_1b_1 = amp_calc
answer_12_1b_2 = amp_theory
```


How can I check though that the frequency of my steady-state oscillations is correct? One way we can do this is by using the Fourier transform.

**Exercise 1(c):** Calculate the power spectrum of the second half of the calculated DDHO response trace calculated in 1(a). Make a plot of the power spectrum with a logarithmic scale on the y-axis, and find the frequency in the data corresponding to the highest spectral power. 

In your power spectrum, keep only the part of the spectrum at positive frequencies.

```python
# Pick out the second half of the trace to analyze for the steady state
# xss = ...
# Tss = ...
# ...
# power = ...
# f = ...

# Now keep only the positive frequencies
end = int(N/4)
f_pow = f[0:end]
power = power[0:end]

# Make a plot

# Find the frequency of the peak
# ...
# fmax = ...
# fmax_expected = ...
# ...

print("Calculated frequency:  %.5f Hz (%.5f Rad/s)" %(f[max_index], f[max_index]*2*np.pi))
print("Theoretical frequency: %.5f Hz (%.5f Rad/s)" % (1/2/np.pi, 1))

answer_12_1c_1 = power
answer_12_1c_2 = f_pow
answer_12_1c_3 = fmax
```

```python
question = "answer_12_1c"
num = 3

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
# Pick out the second half of the trace to analyze for the steady state
# xss = ...
# Tss = ...
# ...
# power = ...
# f = ...
### BEGIN SOLUTION
xss = x[int(N/2):]
Tss = T/2
print(Tss)
fs = 1/(t[1]-t[0])
xss_t = np.fft.fft(xss)
power = 2*np.abs(xss_t)**2/(fs*Tss)**2
f = np.fft.fftfreq(len(xss), 1/fs)
### END SOLUTION

# Now keep only the positive frequencies
end = int(N/4)
f_pow = f[0:end]
power = power[0:end]

# Make a plot
### BEGIN SOLUTION
plt.plot(f_pow,power)
plt.ylabel("Power spectrum of $x$ (m$^2$)")
plt.xlabel("Frequency (Hz)")
plt.yscale('log')
### END SOLUTION

# Find the frequency of the peak
# ...
# fmax = ...
# fmax_expected = ...
# ...
### BEGIN SOLUTION
pmax = 0;
max_index = 0;
for i in range(len(f_pow)):
    if power[i] > pmax:
        pmax = power[i]
        max_index = i
fmax = f[max_index]
### END SOLUTION

print("Calculated frequency:  %.5f Hz (%.5f Rad/s)" %(f[max_index], f[max_index]*2*np.pi))
print("Theoretical frequency: %.5f Hz (%.5f Rad/s)" % (1/2/np.pi, 1))

answer_12_1c_1 = power
answer_12_1c_2 = f_pow
answer_12_1c_3 = fmax
```


For the simple Harmonic oscillator, the steady state solution does not depend on the initial conditions (this is not true for non-linear restoring forces, which we will see soon!). 

**Exercise 1(d):** Perform the calculation from Exercise 1(a) with an initial condition $x(0) = 30$. Make a plot of $x(t)$. Write a function `find_ss_amp(x)` to find the steady state amplitude as we did above in 1(b) by finding the peak value of the last peak. Use this function to find the steady state amplitude and show that this is the same (or at least close) to the value for $x(0) = 0$. 

```python
def find_ss_amp(x):
    # ...

x0 = 30
v0 = 0
# sol = solve_ivp(....)
# t = ...
# x = ...


# Make a plot

print("Calculated steady state amplitude: ", find_ss_amp(x))

answer_12_1d_1 = np.copy(x)
```

```python
question = "answer_12_1d"
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
def find_ss_amp(x):
    ### BEGIN SOLUTION
    peak_indices, _ = find_peaks(x)
    return x[peak_indices[-1]]
    ### END SOLUTION

x0 = 30
v0 = 0
# sol = solve_ivp(....)
# t = ...
# x = ...
### BEGIN SOLUTION
sol = solve_ivp(dydt, (0,T), (x0,v0), t_eval=t)
t = sol.t
x = sol.y[0,:]
### END SOLUTION


# Make a plot
### BEGIN SOLUTION
plt.plot(t,x)
plt.xlabel("Time (s)")
plt.xlabel("Position (m)")
### END SOLUTION

print("Calculated steady state amplitude: ", find_ss_amp(x))

answer_12_1d_1 = np.copy(x)
```


## Boundary value problems and the shooting method

Until now, we have been finding the solutions of differential equations given a particular initial condition: for example, for a second order differential equation, we have been specifying the initial velocity and initial position.

There are some times where you want to constrain your solution of the differential equation not by the initial conditions, but by a combination of the initial condition and the final condition. An example of this is for a second order ODE: instead of finding the solution corresponding to a given initial position $x_i$ and initial velocity $v_i$, one might want to find a solution that has a given initial position $x_i$ and a given final position $x_f$. This is an example of a boundary value problem: we are constrained by the values of the function on the boundaries, not by the initial velocity.

How does one approach such a problem? 

One method for solving this is a technique known as the <a href=https://en.wikipedia.org/wiki/Shooting_method>shooting method</a>. In the shooting method, one makes a guess at an initial velocity $v_i$ and then calculates the final position $x_f$. If you do not get it right in the first guess, you keep redoing the calculation with different $v_i$, using a technique such as <a href=https://en.wikipedia.org/wiki/Binary_search_algorithm>binary search</a> to find the initial velocity $v_i$ that gives the desired $x_f$.

_(The inspiration for the name of this technique comes from how one would hit a target with a cannon: a simple way is to take a trial shot, see where it lands, and then adjust the angle up and down until you get it right; or at least close enough to hit your enemy!)_

As an example, we consider the following problem: I am going to throw a ball straight upwards in the air, and I want the ball to land back down at the same spot in exactly 10 seconds. What velocity do I need to give the ball? 

Neglecting air resistance, this problem is easy to solve (you can solve it with high school physics!). Once we add air resistance to the problem (which we will at the end of this notebook), the problem becomes very difficult to solve analytically, which is where numerical methods become very useful.

We have learned already how to perform RK4 numerical integration of differential equations, and earlier in the course, we have also learned how to find zeros of a function efficiently using binary search. With the skills you have learned, you could easily write code to perform the shooting method directly yourself (and you may be asked to in the exam!). 

However, here, we will explore how to implement the shooting method using some more advanced features of the <a href=https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>solve_ivp()</a> routine of the `scipy` library. To do this, we will make use of the optional parameter `events` of the `solve_ivp()` routine. This is a special way to tell `solve_ivp()` to detect conditions on the integration it is performing and stop the integration when that condition is satisfied. 

The way `events=` parameter works is that you need define a function that gets passed your variables `y` and `t` and which should return a number. When your "event" function undergoes a zero crossing, then you can have `solve_ivp()` stop the integration automatically. You can also configure if this zero crossing is a "rising" or a "falling" edge trigger (see the assignment from week 1). 

How do I use this `events` parameter in practice? The `events` parameter needs to take a <a href=https://docs.python.org/3/tutorial/classes.html>object</a> (or a list of objects in case you want to track many events).

**Wait a minute: what the heck is an "object"?**

Objects are an advanced feature of Python. In fact, most of the things you have been working with are already objects, even though you didn't notice it! Objects are special variables that can "contain" stuff: they can contain, for example, both functions that can perform operations and data that stores values.

An example you have already seen is numpy arrays. For example, consider the following numpy array:

<!-- #raw nbgrader={"grade": false, "grade_id": "cell-68222414656c7a29", "locked": true, "schema_version": 3, "solution": false, "task": false} -->
x = np.array(range(10))

This array has a "field" (data stored inside the object) called `shape` that tells you what shape the array is:

```python nbgrader={"grade": false, "grade_id": "cell-ec2404667f92c47c", "locked": true, "schema_version": 3, "solution": false, "task": false}
x.shape
```

The object `x` also has functions built into it that can excute actions. For example, there is a function `copy` built into `x` that makes a copy of the array:
<!-- #endregion -->

<!-- #raw nbgrader={"grade": false, "grade_id": "cell-234d6036e8a42b7e", "locked": true, "schema_version": 3, "solution": false, "task": false} -->
x2 = x.copy()
<!-- #endraw -->

Cool! But how do I make my own "object"? It sounds scary! In most languages, it requires some more detailed of knowledge of the language to do so... However, python has some great shortcuts that makes building your own objects very easily, on the fly, using a technique in python with a funny name called "<a href=https://www.geeksforgeeks.org/monkey-patching-in-python-dynamic-behavior/>monkey patching</a>".

You can construct the "object" that the `solve_ivp()` parameter `events=` requires by first creating a function that does what you need and then turn it into an "object" by adding some fields with the correct names. To be concrete, for our example of throwing a ball vertically in the air, we will want to stop the numerical integration once the height of the ball falls back down to zero. For this, I will need an event function that returns the height of the ball. If I choose my `y` variable array such that `y[0]` is the vertical position `x` and `y[1]` is the vertical velocity `v`, then my function would look like this:

```
def myevent(t,y):
    return y[0]
```

When you give it this function, the `solve_ivp()` routine will work to find the precise integration conditions and trigger the event such that the value that your function `myevent()` returns the value zero. In this case, because we are looking for the condition that the height of the ball is zero, then life is very easy: we just return the height of the ball! If we instead wanted to trigger the event when the height of the ball is 1, then we would replace the last line of our function with `return y[0]-1`. 

For `solve_ivp()`, I then need to make it an object by adding two extra fields. One field is called `terminal`, which tells `solve_ivp()` if should stop when this event is triggered, or if it should keep on going. As we are done once the ball hits the ground, we can make our event terminal:

```
myevent.terminal = True
```

Magico! Now `myevent` is no longer just a function that can do stuff, but an object that contains a field that stores some information! In addition, we need to add another field to our `myevent` object that tells `solve_ivp()` if it should look for a "rising edge zero crossing" (direction = 1) or a "falling edge zero crossing" (direction = -1). We want the latter:

```
myevent.direction = -1
```

When finished, the `sol` object that `solve_ivp()` returns will contain a field `t_events` that is a list of all the times that an event occurred at: for us, this list will contain one entry, corresponding to the time at which the ball came back to the place we threw it from.

With this, it becomes very easy to implement the shooting method:

1. We define an "event" function that returns the value of the height of the ball
2. We make a guess of an lower and upper limit on the velocity in between which we think the correct initial velocity can be found
3. We use binary search to iterate through initial velocities until we find one that gives us a total time of 10 seconds for the ball fall back down (within a given accuracy). 

One problem with this technique is that we need to get some reasonable values for the initial guesses. For this, it is handy to run the numerical integration a few times with different values of initial velocity to see what range we should choose for our binary search.

**Exercise 2(a):** Consider a ball with mass $m = 0.1$ kg thrown upwards into the air with an initial velocity of 1 m/s. Use `solve_ivp()` to calculate how long it takes for the ball to reach the ground again. Assume that the $g = 9.81$ m/s$^2$. For the numerical integration, you need to pick a time endpoint that will be longer than the time it will take for the ball to come back down. A time limit of 100 seconds should probably be fine.

```python
m=0.1
g = 9.81 

# y[0] is height (which I will call "x"), y[1] is velocity ("v")
def myevent(t,y):
    # ...

    

def dydt(t,y):
    # ...

vi = 1
# sol = solve_ivp(....)

t_f = sol.t_events[0]

print("An initial velocity of 1 m/s gave a total time of %f seconds" % t_f)

answer_12_2a_1 = t_f
```

```python
question = "answer_12_2a"
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
m=0.1
g = 9.81 

# y[0] is height (which I will call "x"), y[1] is velocity ("v")
def myevent(t,y):
    ### BEGIN SOLUTION
    return y[0]
    ### END SOLUTION

    
### BEGIN SOLUTION
myevent.terminal = True
myevent.direction = -1
### END SOLUTION

def dydt(t,y):
    ### BEGIN SOLUTION
    x = y[0]
    v = y[1]
    return [v, -g]
    ### END SOLUTION

vi = 1
# sol = solve_ivp(....)
### BEGIN SOLUTION
sol = solve_ivp(dydt, (0,100), [0,vi], events=myevent)
### END SOLUTION

t_f = sol.t_events[0]

print("An initial velocity of 1 m/s gave a total time of %f seconds" % t_f)

answer_12_2a_1 = t_f
```


**Exercise 2(b):** Implement binary search to find the initial velocity needed to have $t_f = 10$ seconds with a target accuracy of 1 ms. For your initial search range, choose $v_i$ = 1 m/s and 100 m/s. Compare to what you would expect theoretically from the acceleration due to gravity with no air resistance.

```python
def find_tf(vi):
    # ...

v1 = 1
t1 = find_tf(v1)
v2 = 200
t2 = find_tf(v2)
target = 1e-3

# while np.abs(t2-t1) > target:
# ...
    

# vi = ...
# vi_theory = ...

print(vi)
print(vi_theory)

answer_12_2b_1 = vi
answer_12_2b_2 = vi_theory
```

```python
question = "answer_12_2b"
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
def find_tf(vi):
    ### BEGIN SOLUTION
    sol = solve_ivp(dydt, (0,100), [0,vi], events=myevent)
    return sol.t_events[0]
    ### END SOLUTION

v1 = 1
t1 = find_tf(v1)
v2 = 200
t2 = find_tf(v2)
target = 1e-3

# while np.abs(t2-t1) > target:
# ...
### BEGIN SOLUTION
while np.abs(t2-t1) > target:
    vp = (v1+v2)/2
    tp = find_tf(vp)
    if (t1-10)*(tp-10) > 0:
        v1 = vp
        t1 = tp
    else:
        v2 = vp
        t2 = tp
### END SOLUTION
    

# vi = ...
# vi_theory = ...
### BEGIN SOLUTION
vi = (v1+v2)/2
vi_theory = 9.81*5
### END SOLUTION

print(vi)
print(vi_theory)

answer_12_2b_1 = vi
answer_12_2b_2 = vi_theory
```


**Exercise 2(c):** We will now add air resistance to our calculation. To make life simple, we will assume that the <a href=https://en.wikipedia.org/wiki/Drag_coefficient>drag coefficient</a> of our ball results in a friction force with the magnitude:

$$
|F_f| = Cv^2
$$

For a 3 cm diameter sphere, Gary's estimate of the constant $C$ in this equation (based on the formulas on wikipedia) is that it has a value on the order of $C = 10^{-3}$ Ns$^2$/m$^2$. Since the force always opposes the direction of the velocity, we have to somehow account for its sign before incorporating it into an equation of motion. One way to do this is as follows:

$$ 
F_f = - \frac{Cv^3}{|v|}
$$

Repeat question 2(a) but now including air resistance in your calculation.

Before you start: do you think it will take less time or more time for the ball to fall for the same velocity? 

On the one hand, when falling, the ball will move more slowly because the air resistance is holding it back. On the other hand, the ball will not go as high with air resistance. 

(Give your `dydt` function with air resistance a different name so we can compare the two in the next question...)

```python
your_guess = "comes back in less time /OR/ takes more time to come back (delete one!)"

g = 9.81 
m = 0.1
C = 1e-3

# y[0] is height (which I will call "x"), y[1] is velocity ("v")
def myevent(t,y):
    # ...
    

def dydt2(t,y):
    # ...

vi = 1
# sol = solve_ivp(....)
t_f = sol.t_events[0]

print("An initial velocity of 1 m/s gave a total time of %f seconds" % t_f)

answer_12_2c_1 = t_f
```

```python
question = "answer_12_2c"
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
your_guess = "comes back in less time /OR/ takes more time to come back (delete one!)"

g = 9.81 
m = 0.1
C = 1e-3

# y[0] is height (which I will call "x"), y[1] is velocity ("v")
def myevent(t,y):
    ### BEGIN SOLUTION
    return y[0]
    ### END SOLUTION
    
### BEGIN SOLUTION
myevent.terminal = True
myevent.direction = -1
### END SOLUTION

def dydt2(t,y):
    ### BEGIN SOLUTION
    x = y[0]
    v = y[1]
    return [v, -g - C*v**3/np.abs(v)/m]
    ### END SOLUTION

vi = 1
# sol = solve_ivp(....)
### BEGIN SOLUTION
sol = solve_ivp(dydt2, (0,100), [0,vi], events=myevent)
### END SOLUTION
t_f = sol.t_events[0]

print("An initial velocity of 1 m/s gave a total time of %f seconds" % t_f)

answer_12_2c_1 = t_f
```


It seems like adding air resistance slightly decreased the amount of time it takes for the ball to come back down, but not by much. 

**Exercise 2(d):** Make a calculation of the two times as a function of the initial velocity, with $v_i$ varying from 1 m/s to 200 m/s (720 km/h!) with 100 points. Make a plot of $t_f$ vs $v_i$ for the two cases. 

```python
Ni = 200
vi = np.linspace(1,200,Ni)
tf1 = np.empty(Ni)
tf2 = np.empty(Ni)

for i in range(Ni):
    # ... dydt

for i in range(Ni):
    # ... dydt2

# Make some plots

answer_12_2d_1 = vi.copy()
answer_12_2d_2 = tf1.copy()
answer_12_2d_3 = tf2.copy()
```

```python
question = "answer_12_2d"
num = 3

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
Ni = 200
vi = np.linspace(1,200,Ni)
tf1 = np.empty(Ni)
tf2 = np.empty(Ni)

for i in range(Ni):
    ### BEGIN SOLUTION
    sol = solve_ivp(dydt, (0,100), [0,vi[i]], events=myevent)
    tf1[i] = sol.t_events[0]
    ### END SOLUTION

for i in range(Ni):
    ### BEGIN SOLUTION
    sol = solve_ivp(dydt2, (0,100), [0,vi[i]], events=myevent)
    tf2[i] = sol.t_events[0]
    ### END SOLUTION

# Make some plots
### BEGIN SOLUTION
plt.plot(vi, tf1, label="Frictionless")
plt.plot(vi, tf2, label="With air resistance")
plt.xlabel("Initial velocity v$_i$ (m/s)")
plt.ylabel("Total time t$_f$ (s)")
### END SOLUTION

answer_12_2d_1 = vi.copy()
answer_12_2d_2 = tf1.copy()
answer_12_2d_3 = tf2.copy()
```


**Exercise 2(e):** Calculate the initial velocity required to achieve $t_f$ of 10 seconds including air resistance. 

```python
def find_tf2(vi):
    # ...

v1 = 1
t1 = find_tf2(v1)
v2 = 200
t2 = find_tf2(v2)
target = 1e-3

# while np.abs(...) > target:
#     ...

# vi = ...

print(vi)

answer_12_2e_1 = vi
```

```python
question = "answer_12_2e"
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
def find_tf2(vi):
    ### BEGIN SOLUTION
    sol = solve_ivp(dydt2, (0,100), [0,vi], events=myevent)
    return sol.t_events[0]
    ### END SOLUTION

v1 = 1
t1 = find_tf2(v1)
v2 = 200
t2 = find_tf2(v2)
target = 1e-3

# while np.abs(...) > target:
#     ...
### BEGIN SOLUTION
while np.abs(t2-t1) > target:
    vp = (v1+v2)/2
    tp = find_tf2(vp)
    if (t1-10)*(tp-10) > 0:
        v1 = vp
        t1 = tp
    else:
        v2 = vp
        t2 = tp
### END SOLUTION

# vi = ...
### BEGIN SOLUTION
vi = (v1+v2)/2
vi_t = 9.81*5
### END SOLUTION

print(vi)

answer_12_2e_1 = vi
```


**Exercise 2(f):** Including air resistance, does the ball spend more time on the upwards trajectory, on the downward tragectory, or does it spend the same amount of time going up as it does going down?

Use the `solve_ivp()` routine to calculate the time going up and the time going down for the initial velocity you found in 2(e). 

_Hint:_ an easy way to do this is to create an additional `terminal = False` event function that checks when the velocity crosses zero!

```python
# y[0] is height (which I will call "x"), y[1] is velocity ("v")
def myevent(t,y):
    # ...


def myevent2(t,y):
    # ...



# sol = solve_ivp(..., events=[myevent2, myevent])

t1 = sol.t_events[0][0]
t2 = sol.t_events[1][0]
t_down = t2-t1

print("Time up:   %f" % t1)
print("Time down: %f" % (t2-t1))

answer_12_2f_1 = t1
answer_12_2f_2 = t_down
```

```python
question = "answer_12_2f"
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
# y[0] is height (which I will call "x"), y[1] is velocity ("v")
def myevent(t,y):
    ### BEGIN SOLUTION
     return y[0]
    ### END SOLUTION

### BEGIN SOLUTION
myevent.terminal = True
myevent.direction = -1
### END SOLUTION

def myevent2(t,y):
    ### BEGIN SOLUTION
    return y[1]
    ### END SOLUTION

### BEGIN SOLUTION
myevent2.terminal = False
myevent2.direction = -1
### END SOLUTION


# sol = solve_ivp(..., events=[myevent2, myevent])
### BEGIN SOLUTION
sol = solve_ivp(dydt2, (0,200), [0,vi], events=[myevent2, myevent])
### END SOLUTION

t1 = sol.t_events[0][0]
t2 = sol.t_events[1][0]
t_down = t2-t1

print("Time up:   %f" % t1)
print("Time down: %f" % (t2-t1))

answer_12_2f_1 = t1
answer_12_2f_2 = t_down
```


```python tags=["auto-execute-page", "thebe-init", "hide-input"]
## Pre-loading the solutions

import sys
await micropip.install("numpy")
from validate_answers import *

with open(location):
    pass # Initially this notebook does not recognise the file unless someone tries to read it first
```
