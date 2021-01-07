import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# colors for ramachandran plots as in Vitalini, Keller et al. 2015
rmchd_colors = [(171, 30, 34),
                (245, 127, 31),
                (246, 235, 22),
                (105, 188, 69),
                (111, 204, 221),
                (41, 54, 146),
                (255, 255, 255)]

# color scale between 0 and 1
rmchd_colors = list(np.array(rmchd_colors) / 255)[::-1]


def sequential(ax, values, timestep=1, average=None, fmt=None):
    '''
    Plot sequential value (e.g. RMSD)over trajectory.

    Parameters
    ----------
    ax : plt.Axes
        Ax for plotting.
    values : array-like
        Sequential values.
    timestep : int or float
        Timestep of the trajectory in picoseconds. Default 1.
    average : 'expanding', 'moving' or None
        Plot expanding or moving average. Default None.
    fmt : str or None
        Format string for plotting.
    '''
    t_total = len(values) * timestep
    xs = np.arange(0, t_total, timestep)
    ys = values

    # set time unit to ns for long trajectories
    if t_total >= 10000:
        t_unit = 'ns'
        xs *= 0.001
    else:
        t_unit = 'ps'

    # plot values
    if fmt:
        ax.plot(xs, ys, fmt)
        ax.set_xlabel(f't [{t_unit}]')
    else:
        ax.plot(xs, ys)
        ax.set_xlabel(f't [{t_unit}]')

    # plot average
    if average == 'expanding':
        expanding_avg = np.zeros(len(ys))
        for i in range(len(ys)):
            expanding_avg[i] = np.mean(ys[:i])
        ax.plot(xs, expanding_avg, 'r-')
    elif average == 'moving':
        pass


def histogram(ax, values, n_chunks=None, bins=100, density=True):
    '''
    Plot histogram. Use keyword arguments for np.histogram.

    Parameters
    ----------
    ax : plt.Axes
        Ax for plotting.
    values : list or array-like
        Values from which histogram will be calculated and plotted.
    n_chunks : int or None
        If None, plot histogram for all values. If int, separate values into chunks
        and plot average histogram with error bars indicating standard deviation.
    bins : int
        Number of histogram bins.
    density : bool
        Wether to plot probability density.

    '''
    if not n_chunks:
        hist, bin_edges = np.histogram(values, bins=bins, density=density)

        bin_width = bin_edges[1] - bin_edges[0]
        x_bins = bin_edges[1:] - bin_width / 2
        ax.bar(x_bins, hist, bin_width, color='k')

    else:
        if not len(values) // n_chunks == len(values) / n_chunks:
            raise ValueError('Total number of values must be dividable by n_chunks.')

        chunksize = len(values) // n_chunks
        hists = np.zeros((n_chunks, bins))
        means = np.zeros(bins)
        stds = np.zeros(bins)

        for i in range(n_chunks):
            start = i * chunksize
            stop = (i + 1) * chunksize
            hist, bin_edges = np.histogram(
                values[start:stop], bins=bins, density=density, range=(values.min(), values.max()))
            hists[i] = hist

        for i in range(bins):
            means[i] = np.mean(hists[:, i])
            stds[i] = np.std(hists[:, i])

        bin_width = bin_edges[1] - bin_edges[0]
        x_bins = bin_edges[1:] - bin_width / 2

        ax.bar(x_bins, means, bin_width, color='k')
        ax.plot(x_bins, means + stds, color='grey')
        ax.plot(x_bins, means - stds, color='grey')


def ramachandran(ax, phis, psis):
    '''
    Make Ramachandran plot for given phi and psi dihedrals.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib Axes that is plotted on.
    phis : list or array-like
        Phi backbone dihedrals.
    psis : list or array-like
        Psi backbone dihedrals.
    '''
    hist_rc = np.histogram2d(phis, psis, bins=360)[0]
    cm = LinearSegmentedColormap.from_list('rmchd', rmchd_colors, N=7)
    im = ax.pcolormesh(hist_rc, cmap=cm, vmin=hist_rc.min(), vmax=hist_rc.max())

    ax.set(xticks=np.arange(0, 361, 60),
           yticks=np.arange(0, 361, 60),
           xticklabels=np.arange(-180, 181, 60),
           yticklabels=np.arange(-180, 181, 60))
