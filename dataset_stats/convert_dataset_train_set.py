import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
# from ..lib import dirs # Doesn't work on unix, why?

# Train
source_dir = "/home/common/datasets/SIIM-ISIC_2020_Melanoma/jpeg/train/"
dest_dir   = "/home/common/datasets/SIIM-ISIC_2020_Melanoma/jpeg/train_compact/"

# dirs.create_folder(dest_dir)

file_list = glob(source_dir+"*.jpg")
print("Converting image files...")
for file_path in tqdm(file_list):
    file_name = Path(file_path).name
    os.system("convert \""+file_path+"[512x]\" -set filename:base \"%[basename]\" \""+dest_dir+"/%[filename:base].jpg\"")
