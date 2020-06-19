import torch
import torchvision
import numpy  as np
import pandas as pd
from pathlib import Path
import sklearn.metrics
import time
import uuid
from tqdm import tqdm

import lib.dataset
import lib.dirs as dirs


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

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
        output = self.fc3(feature)
        return output

def train_model(model, dataset, batch_size, optimizer, scheduler, epoch_number,
        use_metadata, loss_balance):
    # Create unique identifier for this experiment.
    identifier = uuid.uuid4()

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
        for x in ("train", "val")}

    # Statistics that will be computed later.
    epoch_auc      = {x: np.zeros(epoch_number) for x in ("train", "val")}
    epoch_loss     = {x: np.zeros(epoch_number) for x in ("train", "val")}
    epoch_accuracy = {x: np.zeros(epoch_number) for x in ("train", "val")}
    for i in range(epoch_number):
        print("\nEpoch: {}/{}".format(i + 1, epoch_number))
        for phase in ("train", "val"):
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
            epoch_target = np.concatenate(epoch_target, axis=0)
            epoch_confidence = np.concatenate(epoch_confidence, axis=0)
            epoch_loss[phase][i] /= len(dataset[phase]) #sample_number
            epoch_correct = epoch_target == (epoch_confidence > 0.5)
            epoch_accuracy[phase][i] = (epoch_correct.sum() /
                    len(dataset[phase])) #sample_number
            epoch_auc[phase][i] = sklearn.metrics.roc_auc_score(epoch_target,
                    epoch_confidence)
            epoch_seconds = time.time() - epoch_seconds

            time_string = "{:.0f}h {:.0f}m {:.0f}s".format(epoch_seconds // 3600, \
                epoch_seconds // 60 % 60, epoch_seconds % 60)
            print("Epoch complete in ", time_string)
            print("{} loss: {:.4f}".format(phase, epoch_loss[phase][i]))
            print("{} accuracy: {:.4f}".format(phase, epoch_accuracy[phase][i]))
            print("{} area under ROC curve: {:.4f}".format(phase, epoch_auc[phase][i]))

            results_df = pd.DataFrame({
                                        "target":        epoch_target[i],
                                        "confidence":    epoch_confidence[i],
                                        "loss-train":    epoch_loss['train'][i],
                                        "loss-val":      epoch_loss['val'][i],
                                        "correct":       epoch_correct,
                                        "accuracy-train":epoch_accuracy['train'][i],
                                        "accuracy-val":  epoch_accuracy['val'][i],
                                        "auc-train":     epoch_auc['train'][i],
                                        "auc-val":       epoch_auc['val'][i],
                                        "seconds":       epoch_seconds
            })
        # Save model
        experiment_path = Path(dirs.experiments) / str(identifier)
        weights_path = (experiment_path / "weights") / "resnet18_{}_{}.pth".format(i, identifier)
        dirs.create_folder(weights_path.parent)
        torch.save(model.state_dict, weights_path)
        results_df.to_json("epoch_{}_results.json".format(i))

if __name__ == "__main__":
    pass