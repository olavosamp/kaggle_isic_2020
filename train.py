import torch
import torchvision
import numpy as np

from lib.model      import MetadataModel, train_model
import lib.model
import lib.dataset
import lib.dirs      as dirs
import lib.utils     as utils
import lib.vis_utils as vutils
import lib.defines   as defs

if __name__ == "__main__":
    data_path       = dirs.dataset
    use_metadata    = True
    loss_balance    = True
    batch_size      = 128
    learning_rate   = 0.001
    weight_decay    = 0.0001
    momentum        = 0.9
    epoch_number    = 10
    step_size       = 20
    gamma           = 0.1
    data_sample_size= 1.   # This should be 1 for training with the entire dataset
    freeze_conv     = True
    identifier      = "sample_{:.0f}%_metadata_{}_balance_{}_freeze_{}".format(
                            data_sample_size*100, use_metadata, loss_balance, freeze_conv)

    # Define image transformations
    image_transform = utils.resnet_transforms(defs.IMAGENET_MEAN, defs.IMAGENET_STD)

    # Create train and validation datasets
    dataset = {}
    dataset["train"] = lib.dataset.create_dataset(data_path, "csv/{}_set.csv".format("train"),
                transform=image_transform["train"], balance=True, sample=data_sample_size)
    
    dataset["val"] = lib.dataset.create_dataset(data_path, "csv/{}_set.csv".format("val"),
                transform=image_transform["val"], balance=False, sample=data_sample_size)

    print("Train set size: {}.".format(len(dataset["train"])))
    print("Validation set size: {}.".format(len(dataset["val"])))

    # Load model
    resnet = torchvision.models.resnet18(pretrained=True)
    if use_metadata:
        model_base = torch.nn.Sequential(*list(resnet.children())[:-1])
        model = MetadataModel(model_base, base_out_dim=512)
    else:
        resnet.fc = torch.nn.Linear(512, 2)
        model = resnet
    model.to(lib.model.device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                momentum=momentum, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                gamma=gamma)

    results_folder = train_model(model, dataset, batch_size, optimizer, scheduler, epoch_number,
                        use_metadata, loss_balance=loss_balance, identifier=identifier,
                        freeze_conv=freeze_conv)

    vutils.plot_val_from_results(results_folder, dest_dir=None)
