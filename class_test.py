import os
import numpy as np
from IPython import embed
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.patches import Rectangle

class DataSelector(object):
  #set initialiser
    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0,0),0,0)
        #no rectangle coordinates yet
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion_pressed)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.is_pressed = True

    def on_release(self, event):
        self.x1= event.xdata
        self.y1= event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy(self.x0, self.y0)
        self.ax.figure.canvas.draw()
        self.is_pressed = False

    def on_motion_pressed(self, event):
        if self.is_pressed is True:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy(self.x0, self.y0)
            self.ax.figure.canvas.draw()


fig, ax = plt.subplots()
X = np.random.rand(10)*10
Y=np.random.rand(10)*10
ds = DataSelector()
h,=ax.plot(X,Y,'O')

plt.show()
