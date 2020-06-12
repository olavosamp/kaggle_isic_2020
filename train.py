import torch
import torchvision
import numpy as np

from lib.model import MetadataModel, train_model
import lib.model
import lib.dirs as dirs
import lib.dataset

if __name__ == "__main__":
    data_path = dirs.dataset
    use_metadata = False
    loss_balance = False
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 0.0001
    momentum = 0.9
    epoch_number = 50
    step_size = 20
    gamma = 0.1

    # Define image transformations.
    transform = {"train": torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)),
        ]),
        "val": torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
        ])}
    # Create train and validation datasets.
    dataset = {}
    dataset["train"] = lib.dataset.create_dataset(data_path, "csv/{}_set.csv".format("train"),
                transform=transform["train"], balance=True)
    
    dataset["val"] = lib.dataset.create_dataset(data_path, "csv/{}_set.csv".format("val"),
                transform=transform["val"], balance=False)
    print("Train set size: {}.".format(len(dataset["train"])))
    print("Validation set size: {}.".format(len(dataset["val"])))

    # Load model.
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

    train_model(model, dataset, batch_size, optimizer, scheduler, epoch_number,
            use_metadata, loss_balance)
