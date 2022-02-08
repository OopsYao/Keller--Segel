import matplotlib.pyplot as plt
from solver.fd.operators import DiscreteFunc
import numpy as np
from matplotlib import animation
from tqdm import tqdm
import itertools


def plot(func: DiscreteFunc, **kwargs):
    plt.plot(func.x, func.y, **kwargs)


class Animation:
    '''Animation of function.
    Instance of this class stores frames of functions, then saves it.
    Plotting parameters are given when the instance constructed.'''

    def __init__(self, colors=None, markers='', labels=None, markersize=3,
                 **kwargs):
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.lines = None
        self.f_list = []
        self.t = []

        if not isinstance(colors, (list, tuple)):
            colors = [colors]
        if not isinstance(markers, (list, tuple)):
            markers = [markers]
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        # Cycle color and marker settings for each line
        self.colors = itertools.cycle(colors)
        self.markers = itertools.cycle(markers)
        # Append None to the labels
        self.has_labels = labels.count(None) < len(labels)
        self.labels = itertools.chain(labels, itertools.cycle([None]))

        self.kwargs = kwargs

    def add(self, t, *funcs):
        '''Store values of multiple funcions'''
        self.f_list.append(
            [(np.array(f.x), np.array(f.y)) for f in funcs]
        )
        self.t.append(t)

    def _animate(self, frame):
        t, lx = frame
        if self.lines is None:
            self.lines = [self.ax.plot([], [], **self.kwargs)[0]
                          for _ in range(len(lx))]
            for line, c, m, label in zip(self.lines, self.colors,
                                         self.markers, self.labels):
                if c is not None:
                    line.set_color(c)
                line.set_marker(m)
                if label is not None:
                    line.set_label(label)
        for line, xy in zip(self.lines, lx):
            line.set_data(*xy)
        # Auto scale
        self.ax.relim()
        self.ax.autoscale_view()
        # Title and legend
        self.ax.set_title(t)
        if self.has_labels:
            self.ax.legend()
        return self.lines

    def save(self, filename, desc=None, fps=60, dpi=300, **kwargs):
        frames = list(zip(self.t, self.f_list))
        frames = tqdm(frames, desc=desc)
        ani = animation.FuncAnimation(self.fig, self._animate,
                                      frames=frames)
        ani.save(filename, fps=fps, dpi=dpi, **kwargs)


class Reducer:
    '''Container class that stores a series of items.
    After items outnumber a specific level (volume), only new items that look different
    (from the last item) are stored.'''
    def __init__(self, looks_different: callable, volume: int = 0):
        self.x = None
        self.looks_different = looks_different
        self.counter = 0
        self.container = []
        self.hold_dict = {}
        self.volume = volume

    def significant(self, y):
        if self.x is None or self.looks_different(self.x, y):
            self.x = y
            return True
        else:
            return False

    def add(self, y):
        if self.volume > 0:
            self.container.append(y)
            # Keep the size the container no more than `volume`
            self.container = self.container[-self.volume:]
        if self.x is None or self.looks_different(self.x, y):
            self.hold_dict[self.counter] = y
            self.x = y
        self.counter += 1

    def retrieve(self):
        # Merge container and hold_dict
        for i, item in enumerate(reversed(self.container)):
            self.hold_dict[self.counter - i] = item
        keyList = list(self.hold_dict)
        keyList.sort()
        return [self.hold_dict[k] for k in keyList]
