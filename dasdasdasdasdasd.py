import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from IPython import embed

class DataExtractor(object):
    def __init__(self):
        self.ax = ax1
        self.rect = patches.Rectangle((0,0),0,0, facecolor = 'none', edgecolor = 'red')
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion_pressed)
        self.is_pressed = None
    def on_press(self, event):

        self.x0 = event.xdata
        self.y0 = event.ydata
        self.is_pressed=True
    def on_release(self, event):

        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()
        self.is_pressed=False

        # after selection Plot all selected data in subplot 2
        # if x and Y exist:
        if 'X' and 'Y' in globals():
            # do x and y contain data in selected area x0-x1 an d y0-y1
            self.indexX = [i for i,v in enumerate(X) if v<= self.x1 and v >= self.x0]
            self.indexY = [i for i, v in enumerate(Y) if v <= self.y1 and v >= self.y0]

            selectedX = X[self.indexX]
            selectedY = Y[self.indexY]
            print(self.selectedY)
            print(self.selectedX)
            #plot selected data in subplot 2
            self.ax = ax2
            self.ax.plot(self.selectedX, self.selectedY, marker='o', linestyle='None', color='red')

    def on_motion_pressed(self, event):
        if self.is_pressed is True:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
            self.ax.figure.canvas.draw()

fig, [ax1, ax2] = plt.subplots(1,2)

X = np.random.rand(10)*10
Y= np.random.rand(10)*10
#create rectangle patch

#add patch to plot
#ax.add_patch(rect)

#plot data
test, = ax1.plot(X,Y, marker ='o', linestyle = 'None')

dataextract = DataExtractor()

plt.show()

embed()