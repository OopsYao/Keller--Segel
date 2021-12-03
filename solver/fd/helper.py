import matplotlib.pyplot as plt
from solver.fd.operators import DiscreteFunc
import numpy as np
from matplotlib import animation
from tqdm import tqdm
import itertools
from collections.abc import Iterable


def plot(func: DiscreteFunc, **kwargs):
    plt.plot(func.x, func.y, **kwargs)


class Animation:
    '''Animation of function'''

    def __init__(self, colors=None, markers='', labels=None, markersize=3,
                 **kwargs):
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.lines = None
        self.f_list = []
        self.t = []

        if not isinstance(colors, Iterable):
            colors = [colors]
        if not isinstance(markers, Iterable):
            markers = [markers]
        if not isinstance(labels, Iterable):
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
