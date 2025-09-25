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
plt.figure(figsize=(8,8))
plt.axis('off')

lw =3
plt.plot((1,0),(1,1),'r',lw=lw)
plt.plot((0,0),(0,1),'b',lw=lw)
plt.plot((0,1),(0,0),'b',lw=lw)
plt.plot((1,1),(0,1),'b',lw=lw)

for i in range(11):
    x = i/10
    plt.plot((x,x),(0,1),'k', ls=':')
    plt.plot((0,1),(x,x),'k', ls=':')
    plt.text(x,-0.05,"$x_{%d}$" %i, fontsize=14, ha='center')
    plt.text(-0.05,x,"$y_{%d}$" %i, fontsize=14, ha='center')

plt.text(0.5,1.05,"$\phi = 1$", c='r',fontsize=20, ha='center')
plt.text(0.5,-0.15,"$\phi = 0$", c='b',fontsize=20, ha='center')
plt.text(-0.2,0.5,"$\phi = 0$", c='b',fontsize=20, ha='center')

plt.text(1.15,0.5,"$\phi = 0$", c='b',fontsize=20, ha='center')

for i in range(1,10):
    for j in range(1,10):
        plt.plot(i/10,j/10,'ok')

for i in range(11):
    plt.plot(0,i/10,'ob')
    plt.plot(1,i/10,'ob')
    
for i in range(1,10):
    plt.plot(i/10,0,'ob')
    plt.plot(i/10,1,'or')
    
plt.savefig("box.png", bbox_inches='tight')
```

```python

```
