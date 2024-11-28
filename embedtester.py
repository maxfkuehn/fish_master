from IPython import embed
import random
import matplotlib.pyplot as plt

def pick_scatter_plot():
    # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)

    y = []
    x = []
    z = []
    a = []
    b = []
    n = 10
    for i in range(n):
        y.append(random.randint(3, 9))
        x.append(random.randint(3, 9))
        z.append(random.randint(3, 9))
        b.append(random.randint(3, 9))
        a.append(random.randint(3, 9))

    y.sort()
    x.sort()
    z.sort()
    b.sort()
    a.sort()
    print(y,x)
    print(z,y)
    print(y,a)
    print(x,b)
    def onpick3(event):
        ind = event.artist
        x = ind.get_xdata()
        y = ind.get_ydata()
        print(x,'\n',y)
    fig = plt.figure()
    plt.plot(y, x, picker=True)
    plt.plot(z, y, picker=True)
    plt.plot(y, a, picker=True)
    plt.plot(x, b, picker=True)

    fig.canvas.mpl_connect('pick_event', onpick3)

if __name__ == '__main__':
    pick_scatter_plot()
    plt.show()



