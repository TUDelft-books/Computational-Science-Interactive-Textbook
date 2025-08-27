---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import matplotlib.pyplot as plt
```

```python
def f(x):
    return (x**2/1.5 - x**3/15)/1.5 + 8

plt.figure(figsize=(8,5))
x = np.linspace(-2,10,500)
y = f(x)

x2 = np.linspace(2,8,500)
y2 = f(x2)

plt.plot(x,y,c='k')
plt.fill_between(x2,y2,fc='lightgrey')
plt.plot((2,2), (0,f(2)), c='k')
plt.plot((8,8), (0,f(8)), c='k')

plt.ylim(-2,15)
plt.arrow(0,-2,0,15,head_width=0.4, head_length=1, fc='k')
plt.arrow(-2,0,12,0,head_width=0.8, head_length=0.6, fc='k')
plt.text(-1,13,"$y$", fontsize=18)
plt.text(11,-1,"$x$", fontsize=18)
plt.axis('off')
plt.savefig("integral.png",bbox_inches = 'tight')
```

```python
plt.figure(figsize=(8,5))
x = np.linspace(-2,10,500)
y = f(x)

x2 = np.linspace(2,8,500)
y2 = f(x2)

plt.plot(x,y,c='k')
#plt.fill_between(x2,y2,fc='lightgrey')
plt.plot((2,2), (0,f(2)), c='k')
plt.plot((8,8), (0,f(8)), c='k')

plt.ylim(-2,15)
plt.arrow(0,-2,0,15,head_width=0.4, head_length=1, fc='k')
plt.arrow(-2,0,12,0,head_width=0.8, head_length=0.6, fc='k')
plt.text(-1,13,"$y$", fontsize=18)
plt.text(11,-1,"$x$", fontsize=18)
plt.axis('off')


x3=np.linspace(2,8,4)
y3=f(x3)
plt.plot(x3,y3,'ok')

for i in range(4):
    plt.text(2+i*2,-1.5,"$x_%d$" % i, fontsize=18, ha='center')
    plt.text(2+i*2,f(2+i*2)+1, "$y_%d$" %i, fontsize=18, ha='center')
    plt.plot((2+i*2,2+i*2),(-0.25,0.25), c='k')
    plt.plot((2+i*2,2+i*2),(0,f(2+i*2)), ":", c='k')
    
plt.savefig("discrete.png",bbox_inches = 'tight')
```

```python
plt.figure(figsize=(8,5))
x = np.linspace(-2,10,500)
y = f(x)

x2 = np.linspace(2,8,500)
y2 = f(x2)

plt.plot(x,y,c='k')
#plt.fill_between(x2,y2,fc='lightgrey')
plt.plot((2,2), (0,f(2)), c='k')
plt.plot((8,8), (0,f(8)), c='k')

plt.ylim(-2,15)
plt.arrow(0,-2,0,15,head_width=0.4, head_length=1, fc='k')
plt.arrow(-2,0,12,0,head_width=0.8, head_length=0.6, fc='k')
plt.text(-1,13,"$y$", fontsize=18)
plt.text(11,-1,"$x$", fontsize=18)
plt.axis('off')

x3=np.linspace(2,8,4)
y3=f(x3)
plt.plot(x3,y3,'ok')

for i in range(4):
    i=2*i
    plt.text(2+i,-1.5,"$x_%d$" % (i/2), fontsize=18, ha='center')
    plt.plot((2+i,2+i),(-0.25,0.25), c='k')
    plt.plot((2+i,2+i),(0,f(2+i)), ":", c='k')
    if i<5:
        plt.gca().add_patch(plt.Rectangle((2+i,0),2,f(2+i),fc='lightgrey'))
        
plt.savefig("simple_sum.png",bbox_inches = 'tight')
```

```python
plt.figure(figsize=(8,5))
x = np.linspace(-2,10,500)
y = f(x)

x2 = np.linspace(2,8,500)
y2 = f(x2)

plt.plot(x,y,c='k')
#plt.fill_between(x2,y2,fc='lightgrey')
plt.plot((2,2), (0,f(2)), c='k')
plt.plot((8,8), (0,f(8)), c='k')

plt.ylim(-2,15)
plt.arrow(0,-2,0,15,head_width=0.4, head_length=1, fc='k')
plt.arrow(-2,0,12,0,head_width=0.8, head_length=0.6, fc='k')
plt.text(-1,13,"$y$", fontsize=18)
plt.text(11,-1,"$x$", fontsize=18)
plt.axis('off')


x3=np.linspace(2,8,4)
y3=f(x3)
plt.plot(x3,y3,'ok')

for i in range(4):
    i=2*i
    plt.text(2+i,-1.5,"$x_%d$" % (i/2), fontsize=18, ha='center')
    plt.plot((2+i,2+i),(-0.25,0.25), c='k')
    plt.plot((2+i,2+i),(0,f(2+i)), ":", c='k')
    if i<5:
        xy = ((2+i,0),(2+i,f(2+i)), (4+i,f(4+i)), (4+i,0))
        plt.gca().add_patch(plt.Polygon(xy,fc='lightgrey'))

plt.savefig("trapezoid_rule.png",bbox_inches = 'tight')
```

```python
f= interp1d(x, y, kind='quadratic', fill_value='extrapolate')

```

```python
plt.figure(figsize=(8,5))
x = np.linspace(-2,10,500)
y = f(x)

x2 = np.linspace(2,8,500)
y2 = f(x2)

plt.plot(x,y,c='k')
#plt.fill_between(x2,y2,fc='lightgrey')
plt.plot((2,2), (0,f(2)), c='k')
plt.plot((8,8), (0,f(8)), c='k')

plt.ylim(-2,15)
plt.arrow(0,-2,0,15,head_width=0.4, head_length=1, fc='k')
plt.arrow(-2,0,12,0,head_width=0.8, head_length=0.6, fc='k')
plt.text(-1,13,"$y$", fontsize=18)
plt.text(11,-1,"$x$", fontsize=18)
plt.axis('off')


x3=np.linspace(2,8,4)
y3=f(x3)
plt.plot(x3,y3,'ok')

for i in range(4):
    i=2*i
    plt.text(2+i,-1.5,"$x_%d$" % (i/2), fontsize=18, ha='center')
    plt.plot((2+i,2+i),(-0.25,0.25), c='k')
    plt.plot((2+i,2+i),(0,f(2+i)), ":", c='k')
    if i<5:
        xy = ((2+i,0),(2+i,f(2+i)), (4+i,f(4+i)), (4+i,0))
        plt.gca().add_patch(plt.Polygon(xy,fc='lightgrey'))

plt.savefig("trapezoid_rule.png",bbox_inches = 'tight')
```
