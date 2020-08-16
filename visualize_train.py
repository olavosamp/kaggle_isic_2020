import numpy    as np
from glob       import glob
from pathlib    import Path

import lib.dirs         as dirs
import lib.utils        as utils
import lib.vis_utils    as vutils

if __name__ == "__main__":
    result_path = Path(dirs.experiments) / \
        "sample_100%_metadata_False_balance_True_freeze_False_58ba19e7-d55c-49a9-86c8-29f0d59dd09d"
    image_dir = Path(dirs.images) / result_path.name

    results_df = utils.get_epochs_results(result_path)

    print(results_df)
    mask = results_df.loc[:, 'phase'] == 'val'
    plot_data = results_df[mask]
    
    for ext in ['jpg', 'pdf']:
        vutils.plot_model_history(plot_data['auc'], xlabel="Epochs", ylabel="AUC",
                                    title="Validation Set History",
                                    save_path=image_dir/("model_val_auc."+ext), show=False)
