import os
import time
import uuid
from pathlib import Path
import torch
import torchvision
import numpy  as np
import pandas as pd
import sklearn.metrics
from tqdm import tqdm

import lib.dataset
import lib.dirs as dirs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MetadataModel(torch.nn.Module):
    def __init__(self, base_model, base_out_dim, metadata_dim=9,
            hidden_dim=32):
        super(MetadataModel, self).__init__()
        self.base_model = base_model
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(base_out_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(metadata_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 2)


    def forward(self, image, metadata):
        feature_image = self.base_model(image)
        feature_image = feature_image.view(feature_image.size(0), -1)
        feature_image = self.fc1(feature_image)
        feature_image = self.relu(feature_image)
        feature_metadata = self.fc2(metadata)
        feature_metadata = self.relu(feature_metadata)
        feature = feature_image + feature_metadata
        output  = self.fc3(feature)
        return output


    def freeze_convolutional(self, freeze):
        '''
            Freezes or unfreezes the model convolutional layers.
            Argument:
                freeze: bool
                    Pass True to freeze and False to unfreeze layers.
        '''
        assert type(freeze) == bool, "Argument must be a boolean."

        for layer in self.base_model.children():
            for param in layer.parameters():
                # Freeze == True sets requires_grad = False: freezes layers
                param.requires_grad = not(freeze) 


def freeze_convolutional_resnet(model, freeze):
    '''
        Freezes or unfreezes the model convolutional layers. Assumes the passed model
        has a method named fc that corresponds to the only layer the user wishes to
        keep unfrozen.
        Argument:
            freeze: bool
                Pass True to freeze and False to unfreeze layers.
    '''
    # Freeze (or not) convolutional layers
    for child in model.children():
        for param in child.parameters():
            # Freeze == True sets requires_grad = False: freezes layers
            param.requires_grad = not(freeze)
    
    # Do not freeze FC layer
    for param in model.fc.parameters():
        param.requires_grad = True


def load_model(model, weight_path, device=device, eval=True):
    '''
        Load model weights in corresponding model object.

        model: instantiated model object
            Model into which load the weights.
        
        weight_path: string filepath
            Filepath of the weights.
    '''
    weight_path = str(weight_path)
    assert os.path.isfile(weight_path), "Invalid weight filepath."
    checkpoint = torch.load(weight_path)#, map_location=device)
    # print(checkpoint)
    # input()
    
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint)
    return model


def predict(model, dataloader, device=device, threshold=None, use_metadata=True):
    '''
        Performs inference on the input data using a Pytorch model.
        model: model object
            Model which will perform inference on the data. Must support
            inference through syntax
                 result = model(input)
        data: collection of model inputs
            A collection, such as a list, of model inputs. Each element of data
            must match the required arguments of model.

        threshold: None or float (optional)
            Decision threshold to evaluate model outputs.
            
            If None, result will be an array of floats representing model confidence
            for each input.
             
            If float, result will be an array of zeros and ones. Each input will be
            converted to one if model confidence is greater than threshold and zero
            if lesser.

        Returns:
        result: array of float
    '''
    model.to(device)
    model.eval()
    # if metadata:
    #     def inference_single_input(single_input):
    #         # image = single_input[0].view(1, -1)
    #         image = image.to(device)
    #         metadata = single_input[1].to(device)
    #         return model(image, metadata)
    # else:
    #     def inference_single_input(single_input):
    #         return model(single_input.to(device))

    softmax = torch.nn.Softmax(dim=1)
    result_list = []
    for image, metadata in tqdm(dataloader):
        image    = image.to(device)
        metadata = metadata.to(device)

        if use_metadata:
            output = model(image, metadata)
        else:
            output = model(image)

        confidence = softmax(output).detach().cpu().numpy()[:, 1]
        result_list.append(confidence)

    return result_list


def train_model(model, dataset, batch_size, optimizer, scheduler, num_epochs,
                use_metadata, loss_balance=True, identifier=None,
                freeze_conv=False):
    # Create unique identifier for this experiment.
    if identifier is None:
        identifier = str(uuid.uuid4())
    else:
        identifier = str(identifier) + "_" + str(uuid.uuid4())
    phase_list = ("train", "val")

    # Setup experiment paths
    experiment_dir = Path(dirs.experiments) / str(identifier)
    weights_folder  = experiment_dir / "weights" 
    dirs.create_folder(weights_folder)

    # TODO: Freeze conv layers in a more generic way. Current two function
    # approach is unelegant.
    if use_metadata:
        assert type(model) == MetadataModel, "Model must be a MetadataModel object"
        model.freeze_convolutional(freeze_conv)
    else:
        freeze_convolutional_resnet(model, freeze_conv)

    print("Using device: ", device)

    # Instantiate loss and softmax.
    if loss_balance:
        weight = [1.0, dataset["train"].imbalance_ratio()]
        weight = torch.tensor(weight).to(device)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    # Define data loaders.
    data_loader = {x: torch.utils.data.DataLoader(dataset[x],
        batch_size=batch_size, shuffle=True, num_workers=4)
        for x in phase_list}

    # Measures that will be computed later.
    tracked_metrics = ["epoch", "phase", "loss", "accuracy", "auc", "seconds"]
    epoch_auc       = {x: np.zeros(num_epochs) for x in phase_list}
    epoch_loss      = {x: np.zeros(num_epochs) for x in phase_list}
    epoch_accuracy  = {x: np.zeros(num_epochs) for x in phase_list}
    results_df      = pd.DataFrame()

    for i in range(num_epochs):
        print("\nEpoch: {}/{}".format(i+1, num_epochs))
        results_dict = {metric: [] for metric in tracked_metrics}
        for phase in phase_list:
            print("\n{} phase: ".format(str(phase).capitalize()))

            # Set model to training or evalution mode according to the phase.
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_target = []
            epoch_confidence = []
            epoch_seconds = time.time()
            # Iterate over the dataset.
            for image, metadata, target  in tqdm(data_loader[phase]):
                # Update epoch target list to compute AUC(ROC) later.
                epoch_target.append(target.numpy())

                # Load samples to device.
                image = image.to(device)
                if use_metadata:
                    metadata = metadata.to(device)
                target = target.to(device)

                # Set gradients to zero.
                optimizer.zero_grad()

                # Calculate gradients only in the training phase.
                with torch.set_grad_enabled(phase=="train"):
                    if use_metadata:
                        output = model(image, metadata)
                    else:
                        output = model(image)
                    loss = cross_entropy_loss(output, target)
                    confidence = softmax(output).detach().cpu().numpy()[:, 1]

                    # Backward gradients and update weights if training.
                    if phase=="train":
                        loss.backward()
                        optimizer.step()

                # Update epoch loss and epoch confidence list.
                epoch_loss[phase][i] += loss.item() * image.size(0)
                epoch_confidence.append(confidence)

            if phase == "train":
                scheduler.step()

            # Compute epoch loss, accuracy and AUC(ROC).
            sample_number = len(dataset[phase])
            epoch_target = np.concatenate(epoch_target, axis=0)
            epoch_confidence = np.concatenate(epoch_confidence, axis=0) # List of batch confidences
            epoch_loss[phase][i] /= sample_number
            epoch_correct = epoch_target == (epoch_confidence > 0.5)
            epoch_accuracy[phase][i] = (epoch_correct.sum() / sample_number)
            epoch_auc[phase][i] = sklearn.metrics.roc_auc_score(epoch_target,
                                                                epoch_confidence)
            epoch_seconds = time.time() - epoch_seconds

            time_string   = time.strftime("%H:%M:%S", time.gmtime(epoch_seconds))
            print("Epoch complete in ", time_string)
            print("{} loss: {:.4f}".format(phase, epoch_loss[phase][i]))
            print("{} accuracy: {:.4f}".format(phase, epoch_accuracy[phase][i]))
            print("{} area under ROC curve: {:.4f}".format(phase, epoch_auc[phase][i]))

            # Collect metrics in a dictionary
            results_dict["epoch"].append(i+1) # Epochs start at 1
            results_dict["phase"].append(phase)
            results_dict["loss"].append(epoch_loss[phase][i])
            results_dict["accuracy"].append(epoch_accuracy[phase][i])
            results_dict["auc"].append(epoch_auc[phase][i])
            results_dict["seconds"].append(epoch_seconds)

        # Save metrics to DataFrame
        results_df = results_df.append(pd.DataFrame(results_dict), sort=False, ignore_index=True)

        # Save model
        weights_path = weights_folder / "resnet18_epoch_{}_{}.pth".format(i+1, identifier)
        results_path = experiment_dir / "epoch_{}_results.json".format(i+1)
        torch.save(model.state_dict(), weights_path)
        results_df.to_json(results_path)

    return results_path.parent

if __name__ == "__main__":
    pass
