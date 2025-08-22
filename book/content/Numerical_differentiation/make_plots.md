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
    return (x-1)**2/7 + 2

def plot_vert(x,c):
    plt.plot((x,x), (0,f(x)), c=c)

plt.figure(figsize=(24,8))

x0=5


def plot_slope(x1,x2):
    x = np.linspace(-2,10,500)
    plt.plot(x,f(x),c='k')
    #y=ax+b
    a = (f(x2)-f(x1))/(x2-x1)
    b = f(x2) - a*x2
    plt.plot((0,10),(b, a*10+b), c='k', ls=':')
    plt.ylim(-2,15)
    plt.arrow(0,-2,0,15,head_width=0.5, head_length=0.8, fc='k')
    plt.arrow(-2,0,12,0,head_width=0.6, head_length=0.6, fc='k')
    plt.text(-1,13,"$y$", fontsize=18)
    plt.text(11,-1,"$x$", fontsize=18)
    plt.axis('off')
    plt.plot([x1,x2],[f(x1),f(x2)], 'ok',ms=10)

s=20

plt.subplot(131)
plot_slope(2,5)
plot_vert(5,c='k')
plot_vert(2,c='k')
plt.text(5,-1,"$x_0$",ha='center',fontsize=s)
plt.text(2,-1,"$x_0-h$",ha='center',fontsize=s)
plt.text(4.5,11,"Backward\nDifference",ha='center',fontsize=25)

plt.subplot(132)
plot_slope(5,8)
plot_vert(5,c='k')
plot_vert(8,c='k')
plt.text(5,-1,"$x_0$",ha='center',fontsize=s)
plt.text(8,-1,"$x_0+h$",ha='center',fontsize=s)
plt.text(4.5,11,"Forward\nDifference",ha='center',fontsize=25)


plt.subplot(133)
plot_slope(3.5,6.5)
plt.plot((5,5), (0,f(5)), c='k', ls='--')
plot_vert(3.5,c='k')
plot_vert(6.5,c='k')
plt.text(3.5,-1.4,r"$x_0-\dfrac{h}{2}$",ha='center',fontsize=s)
plt.text(6.5,-1.4,r"$x_0+\dfrac{h}{2}$",ha='center',fontsize=s)
plt.text(4.5,11,"Center\nDifference",ha='center',fontsize=25)

plt.savefig("finite_difference.png", bbox_inches='tight')
```

```python

```
