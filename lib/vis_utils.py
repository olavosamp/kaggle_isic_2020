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
import lib.utils               as utils


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

    x = range(1, len(data)+1)
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


def plot_val_from_results(results_folder, dest_dir=None):
    results_folder = Path(results_folder)
    
    if dest_dir == None:
        dest_dir = Path(dirs.images) / results_folder.name
    else:
        dest_dir = Path(dest_dir)

    results_df = utils.get_epochs_results(results_folder)
    print(results_df)
    mask = results_df.loc[:, 'phase'] == 'val'
    plot_data = results_df[mask]

    for ext in ['jpg', 'pdf']:
        plot_model_history(plot_data['auc'], xlabel="Epochs", ylabel="AUC",
                                    title="Validation Set History",
                                    save_path=dest_dir/("model_val_auc."+ext), show=False)
