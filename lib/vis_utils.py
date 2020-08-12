import os
import matplotlib as mpl
# Use non interactive backend if not running on Windows, meaning it's on the remote server
if os.name != "nt":
    mpl.use("Agg")

import numpy                    as np
import matplotlib.pyplot        as plt
import seaborn                  as sns
from pathlib                    import Path

import lib.dirs                as dirs
import lib.defines             as defs


def set_mpl_fig_options(figsize=defs.MPL_FIG_SIZE_SMALL):
    return plt.figure(figsize=figsize, tight_layout=True)


def plot_model_history(data, data_labels="", xlabel="", ylabel="", title="Model History",
                       save_path=None, show=False):
    '''
        Arguments:
            data: list-like
                A list of numeric values.
            data_labels: string
                String label.
    '''
    assert hasattr(data, "__len__") and hasattr(data, "__iter__"),\
                    "Argument 'data'  must be a list of values or a list of datasets."
    fig = set_mpl_fig_options(figsize=defs.MPL_FIG_SIZE)

    x = range(len(data))
    plt.plot(x, data, '.-', label=data_labels)

    if data_labels != "":
        plt.legend()
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    if save_path is not None:
        save_path = Path(save_path)
        dirs.create_folder(save_path.parent)

        fig.savefig(save_path.with_suffix('.png'),
            orientation='portrait', bbox_inches='tight')

        fig.savefig(save_path.with_suffix('.pdf'),
            orientation='portrait', bbox_inches='tight')

    if show and mpl.get_backend() != "agg":
        plt.show()
    return fig
