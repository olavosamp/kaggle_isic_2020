import os
import matplotlib as mpl
# Use non interactive backend if not running on Windows, meaning it's on the remote server
if os.name != "nt":
    mpl.use("Agg")

import numpy                    as np
import matplotlib.pyplot        as plt
import seaborn                  as sns
from pathlib                    import Path

import .dirs                as dirs
import .defines             as defs

def set_mpl_fig_options(figsize=defs.MPL_FIG_SIZE_SMALL):
    return plt.figure(figsize=figsize, tight_layout=True)


def plot_model_history(data, data_labels=[], xlabel="", ylabel="", title="Model History",
                       save_path=None, show=False):
    '''
        Arguments:
            data: list
            One or more datasets that are lists of y values. If more than one, all datasets
            must be of same length.
            data_labels: string or list of strings
            Data label for each data set given.
    '''
    assert hasattr(data, "__len__"), "Argument 'data'  must be a list of values or a list of datasets."
    fig = set_mpl_fig_options(figsize=defs.MPL_FIG_SIZE)

    # User passed several datasets
    if hasattr(data[0], "__len__"):
        dataLen   = len(data)

        if data_labels == []:
            data_labels = [""]*dataLen
        labelsLen = len(data_labels)

        assert dataLen == labelsLen, "You must pass one label for each dataset"

        x = range(len(data[0]))
        for y, label in zip(data, data_labels):
            # Plot dataset with its corresponding label
            plt.plot(x, y, '.-', label=label)
    # User passed only one dataset
    else:
        x = range(len(data))
        plt.plot(x, data, '.-', label=data_labels)

    plt.legend()
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    if save_path is not None:
        fig.savefig(save_path, orientation='portrait', bbox_inches='tight')
    if show and mpl.get_backend() != "agg":
        plt.show()
    return fig
