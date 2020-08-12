import os
import json
import pandas       as pd
from pathlib        import Path
from glob           import glob
from torchvision    import transforms

# Model
def resnet_transforms(mean, std):
    '''
        Define default transforms for Resnet neural network.
    '''
    dataTransforms = {
            'train': transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
        ]),
            'val': transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
        ])}
    return dataTransforms


# File system
def file_exists(x):
    return Path(x).is_file()


def read_results_json(results_path):
    if file_exists(results_path):
        return pd.read_json(results_path)
    else:
        raise FileNotFoundError("Results file not found.")


def replace_symbols(stringList, og_symbol="\\", new_symbol="/"):
    '''
        Replaces symbols in strings or string lists.
        Replace all instances of og_symbol with new_symbol.
        Defaults to replacing backslashes ("\\") with foward slashes ("/").
    '''
    def _func_replace(x): return str(x).replace(og_symbol, new_symbol)

    if isinstance(stringList, str):         # Input is string
        return _func_replace(stringList)
    elif hasattr(stringList, "__iter__"):   # Input is list of strings
        return list(map(_func_replace, stringList))
    else:
        raise TypeError("Input must be a string or list of strings.")


def get_file_list(folder_path, ext_list=['*'], remove_dups=True, recursive=True):
    '''
        Returns list of files in the file tree starting at folder_path as pathlib.Path objects.
        Optional argument ext_list defines list of recognized extensions, case insensitive.
        ext_list must be a list of strings, each defining an extension, without dots.
        
        Argument remove_dups should almost always be True, else it will return duplicated entries
        as the script searches for both upper and lower case versions of all given extensions.
    '''
    folder_path = str(folder_path)
    assert os.path.isdir(folder_path), "folder_path is not a valid directory."

    # Also search for upper case formats for Linux compatibility
    ext_list.extend([x.upper() for x in ext_list])
    # TODO: Replace this workaround by making a case insensitive search or
    # making all paths lower case before making comparisons (possible?)

    folder_path = replace_symbols(folder_path, og_symbol="\\", new_symbol="/")

    # Add recursive glob wildcard if recursive search is requested
    recurseStr = ""
    if recursive:
        recurseStr = "/**"

    file_list = []
    for extension in ext_list:
        globString = folder_path + recurseStr + "/*."+extension
        globList = glob(globString, recursive=recursive)
        file_list.extend(globList)
    
    if remove_dups:
        # Remove duplicated entries
        file_list = list(dict.fromkeys(file_list))
    
    # file_list = list(map(replace_symbols, file_list))
    # file_list = make_path(file_list)
    return file_list


def get_epochs_results(experiment_folder):
    results_files = get_file_list(experiment_folder, ext_list=["json"], recursive=False)
    # results_dfs = [read_results_json(x) for x in results_files]
    results_files = sorted(results_files)
    results_df = read_results_json(results_files[-1])
    return results_df
