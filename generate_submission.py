import torch
import torchvision
import numpy as np
import pandas as pd
from pathlib import Path

import lib.dataset
import lib.dirs      as dirs
import lib.utils     as utils
import lib.vis_utils as vutils
import lib.defines   as defs
from lib.model      import MetadataModel, load_model, predict

data_path    = dirs.test_set
csv_path     = Path(dirs.csv) / "test.csv"
weight_path  = Path(dirs.experiments) / \
    "sample_100%_metadata_True_balance_True_freeze_False_e5390245-6df2-4fb9-9e67-007bc3f690dd/weights/resnet18_epoch_30_sample_100%_metadata_True_balance_True_freeze_False_e5390245-6df2-4fb9-9e67-007bc3f690dd.pth"
use_metadata = True
data_sample_size = 1.
batch_size = 32

# Test set transform is the same as used in validation set
image_transform = utils.resnet_transforms(defs.IMAGENET_MEAN, defs.IMAGENET_STD)

dataset = lib.dataset.create_test_set(data_path, csv_path,
            transform=image_transform["val"], balance=False, sample=data_sample_size)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=False, num_workers=1)

print("Test set size: {}.".format(len(dataset)))

# Load model
resnet = torchvision.models.resnet18(pretrained=True)
if use_metadata:
    model_base = torch.nn.Sequential(*list(resnet.children())[:-1])
    model = MetadataModel(model_base, base_out_dim=512)
else:
    resnet.fc = torch.nn.Linear(512, 2)
    model = resnet
# model.to(lib.model.device)

model = load_model(model, weight_path)

results = predict(model, data_loader, use_metadata=use_metadata)

results = np.concatenate(results, axis=0)

test_csv = pd.read_csv(csv_path)
test_csv = test_csv.loc[int(len(test_csv)*data_sample_size) ,:]

submission_csv = pd.DataFrame({'image_name': test_csv['image_name'],
                               'target': results})

print(submission_csv.head())
submission_csv.to_csv(dirs.csv / "submission.csv", index=False)
