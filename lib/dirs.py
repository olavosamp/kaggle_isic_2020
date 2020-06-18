import os

dataset = "/home/common/datasets/SIIM-ISIC_2020_Melanoma/jpeg/train/"
weights = "data/weights/"

def create_folder(path, verbose=False):
    if not(os.path.isdir(path)):
        os.makedirs(path)
