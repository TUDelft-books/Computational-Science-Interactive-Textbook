# # ipympl version

# +
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import AppLayout, FloatSlider
import ipywidgets as widgets
import matplotlib.colors as colors

print("Make sure that you have a cell that runs '%matplotlib widget' in your notebook")

#if True: 
# Note: only works right now for square matrices A...but ok for us. 
def explore_image(A, zname="$\phi$", xname="x", yname="y"):
    plt.ioff()
    
    lc = widgets.IntSlider(min=0, max=A.shape[1]-1, description="Linecut:")
    gamma = widgets.FloatLogSlider(min=-1,max=1,step=0.01, description="Color tweak:")
    autoscale = widgets.Checkbox(value=True, description="Autoscale linecut")
    horizontal = widgets.Checkbox(value=True, description="Horizontal linecut")
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3.5))
    ax0, ax1 = axes.flatten()
    
    data_max = np.max(A)
    data_min = np.min(A)
    plot_max = data_max + (data_max - data_min)*0.05
    plot_min = data_min - (data_max - data_min)*0.05
    
    norm = colors.PowerNorm(gamma=gamma.value, vmin=data_min, vmax=data_max)
    im = ax0.imshow(A,cmap='RdBu_r', origin='lower',aspect='auto', norm=norm)
    ax0.set_xlabel(xname)
    ax0.set_ylabel(yname)
    l1 = ax0.plot((0,lc.max),(lc.value,lc.value),c='w', ls=':')        
    
    x = range(lc.max+1)
    l2 = ax1.plot(x,A[lc.value,:])
    ax1.plot((0,lc.max), (0,0), "k--", lw=0.5)
    ax0.set_xlabel(xname)
    
    def update_lines(change):
        if horizontal.value:
            l1[0].set_data((0,lc.max),(lc.value,lc.value))
            l2[0].set_data(x,A[lc.value,:])
        else:
            l1[0].set_data((lc.value,lc.value),(0,lc.max))
            l2[0].set_data(x,A[:,lc.value])
        if autoscale.value:
            ax1.relim()
            ax1.autoscale_view()
        else:
            ax1.set_ylim(plot_min, plot_max)
            ax1.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def update_cmap(change):
        norm = colors.PowerNorm(gamma=gamma.value, vmin=data_min, vmax=data_max)
        im.set_norm(norm)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
    lc.observe(update_lines, names="value")
    gamma.observe(update_cmap, names="value")
    autoscale.observe(update_lines, names="value")
    horizontal.observe(update_lines, names="value")

    controls = widgets.VBox([
        widgets.HBox([lc,gamma]),
        widgets.HBox([autoscale,horizontal])])
    app = AppLayout(center=fig.canvas, footer=controls)

    return app
# -
# a = np.linspace(-10,10,100)
# x,y = np.meshgrid(a,a)
# A = 1/(x**2+y**2+10)*(x+3-y)
#
# explore_image(A)

# # testing two apps at the same time: it works fine
# a = np.linspace(-10,10,100)
# x,y = np.meshgrid(a,a)
# A = 1/(x**2+y**2+10)*(x+3-y)*x
#
# explore_image(A)
