import os

dataset     = "/home/common/datasets/SIIM-ISIC_2020_Melanoma/jpeg/train_compact/"
test_set    = "/home/common/datasets/SIIM-ISIC_2020_Melanoma/jpeg/test_compact/"
csv         = "./csv/"
weights     = "data/weights/"
experiments = "data/experiments/"
images      = "../images/"

def create_folder(path, verbose=False):
    if not(os.path.isdir(path)):
        os.makedirs(path)
