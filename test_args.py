import sys
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import copy

from sklearn.model_selection import train_test_split

from PIL import Image
import numpy as np
import pandas as pd
import os
import re
import cv2


train_data = sys.argv[1]
if os.path.exists(train_data):
    train_labels_file = os.path.join(train_data, "train_label.csv")
    train_images = os.path.join(train_data, "train_images")
else:
    print("Training path not Valid")

test_data = sys.argv[2]
if os.path.exists(test_data):
    test_images = os.path.join(test_data, "test_images")
else:
    print("Test path not Valid")

#################################################################################################
"""
    Helper function to get full path to Images in Train dataset
"""


def get_str_with_path(image, path=train_images):
    image = str(image) + ".png"
    return os.path.join(train_images, image)


#################################################################################################
"""
    Here the train_label.csv is loaded into a dataframe
    and we perform a train-validation split of 80%-20% respectively.
"""
df = pd.read_csv(train_labels_file)
df["ImageName_ID"] = None
df.ImageName_ID = df.ID.apply(lambda x: get_str_with_path(x))
df.index = df.ImageName_ID
df.drop(["ID", "ImageName_ID"], axis=1, inplace=True)
train, val = train_test_split(df, test_size=0.2)
#################################################################################################
"""
    In the first part here, we are trying to get the average histogram bins, values.
    We will use this as the mean histogram values for each bin and transform the images
    to the mean histogram configuration using the custom transform
"""

mean_hist = np.zeros(256)
mean_bins = np.arange(256)


for y, i in enumerate(train.index):
    img = cv2.imread(i, 0)

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.reshape(256)
    mean_hist += hist


mean_hist = mean_hist / (y)
mean_img = (mean_bins, mean_hist)

#################################################################################################
"""
  We are performing histogram matching [ON TRAIN DATA] as a means of image augmentation.
  For more details on what histogram matching is done, please refer to this
  stackoverflow link given below. We have used a variation of it over 3 channels 
  (since the given images have 3 channnels)
  https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
"""


class histogramMatcher(object):
    def __init__(self):
        pass

    def __call__(self, sourceimg):
        sourceimg = np.asarray(sourceimg)
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """
        channels = sourceimg.shape[2]
        out = np.zeros_like(sourceimg)

        for i in range(channels):
            source = sourceimg[:, :, i]
            oldshape = source.shape
            source = source.ravel()

            # get the set of unique pixel values and their corresponding indices and
            # counts
            s_values, bin_idx, s_counts = np.unique(
                source, return_inverse=True, return_counts=True
            )
            t_values, t_counts = mean_img

            # take the cumsum of the counts and normalize by the number of pixels to
            # get the empirical cumulative distribution functions for the source and
            # template images (maps pixel value --> quantile)
            s_quantiles = np.cumsum(s_counts).astype(np.float64)
            s_quantiles /= s_quantiles[-1]
            t_quantiles = np.cumsum(t_counts).astype(np.float64)
            t_quantiles /= t_quantiles[-1]

            # interpolate linearly to find the pixel values in the template image
            # that correspond most closely to the quantiles in the source image
            interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
            out[:, :, i] = interp_t_values[bin_idx].reshape(oldshape)

        return Image.fromarray(out)

    def __repr__(self):
        return self.__class__.__name__ + "()"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#################################################################################################
"""
    Defining data_transforms that will be applied to the respective
    set. We use histogram matcher as the only image augmentation technique
    in our current model work.
"""

data_transforms = {
    "train": transforms.Compose(
        [
            histogramMatcher(),
            transforms.ToTensor(),
        ]
    ),
    "val": transforms.Compose(
        [
            histogramMatcher(),
            transforms.ToTensor(),
        ]
    ),
    "test": transforms.Compose(
        [
            histogramMatcher(),
            transforms.ToTensor(),
        ]
    ),
}
#################################################################################################
"""
    Using general method definition to define Train_Data
    for using in dataloaders to model
"""


class Train_Data(Dataset):
    def __init__(self, labeled_df, transform=None):
        self.labels = list(labeled_df["Label"])
        self.labeled_images = labeled_df
        self.transform = transform

    def __len__(self):
        return len(self.labeled_images)

    def __getitem__(self, idx):
        # apply transforms
        image_path = self.labeled_images.index[idx]

        img = Image.open(image_path)
        img.load()
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, 2)
        image = Image.fromarray(img)

        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return {"image": image, "label": label}


#################################################################################################
"""
    Using general method definition to define Test_Data
    for using in dataloaders to model
"""


class Test_Data(Dataset):
    def __init__(self, image_path, transform=None):

        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        sorted_dir = sorted(os.listdir(image_path), key=alphanum_key)

        image_paths = list()
        for path in sorted_dir:
            if path != ".DS_Store":
                full_image_path = os.path.join(image_path, path)
                image_paths.append(full_image_path)
        self.image_paths = image_paths

        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # apply transforms
        image_path = self.image_paths[idx]

        img = Image.open(image_path)
        img.load()
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, 2)
        image = Image.fromarray(img)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)

        return {"image": image}


#################################################################################################
"""
    Using general method definition to define train_validate_model
    with the given params.
    This function performs the actual training and optimization
    on the given 'model', 'optimizer' for the given 'criterion' and
    'epochs'. This function serves the purpose of a template and allows
    for easy switch between models/optimizers/criterion.
"""


def train_validate_model(model, device, dataloaders, optimizer, criterion, epochs):
    best_model_wts = copy.deepcopy(model.state_dict())

    train_losses_main = []
    val_losses_main = []

    for epoch in range(1, epochs + 1):
        print("Epoch {}/{}".format(epoch, epochs))
        print("-" * 10)

        model.train()
        train_losses = []
        for idx, batch in enumerate(dataloaders["train"]):
            data, target = batch["image"].to(device), batch["label"].long().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if idx % 4 == 0:
                print(
                    "Batches: {}/{}, Loss: {}".format(
                        idx, len(dataloaders["train"]), loss.item()
                    )
                )
        train_loss = torch.mean(torch.tensor(train_losses))
        print("Average Training loss: {:.4f}".format(train_loss))

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for _, batch in enumerate(dataloaders["val"]):
                data, target = batch["image"].to(device), batch["label"].long().to(
                    device
                )
                output = model(data)

                # compute the batch loss
                batch_loss = criterion(output, target).item()
                val_loss += batch_loss
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # divide by the number of batches of batch size 32
        # get the average validation over all bins
        val_loss /= len(dataloaders["val"])
        print(
            "Validation loss: {:.4f},Validation Accuracy: {:.0f}%".format(
                val_loss, 100.0 * correct / len(dataloaders["val"].dataset)
            )
        )

        if (len(val_losses_main) > 0) and (val_loss < min(val_losses_main)):
            torch.save(model.state_dict(), "./ckp/best_model.pt")
            best_model_wts = copy.deepcopy(model.state_dict())
            print(
                "Saving model (epoch {}) with lowest validation loss: {}".format(
                    epoch, val_loss
                )
            )

        train_losses_main.append(train_loss)
        val_losses_main.append(val_loss)
    #             scheduler.step()

    model.load_state_dict(torch.load("./ckp/best_model.pt"))
    return model, train_losses_main, val_losses_main


#################################################################################################
"""
    Defining the different sets and dataloaders for the model
"""

train_set = Train_Data(train, data_transforms["train"])
val_set = Train_Data(val, data_transforms["val"])
test_set = Test_Data(test_images, data_transforms["test"])

batch_size = 16

dataloaders = {
    "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    "val": DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0),
    "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0),
}
criterion = nn.CrossEntropyLoss()
#################################################################################################
"""
    Defining the model for densenet.
    Using Adam as an optimizer for the same
"""
model_densenet = models.densenet201(pretrained=True)
num_ftrs_densenet = model_densenet.classifier.in_features
model_densenet.classifier = nn.Linear(num_ftrs_densenet, 3)
model_densenet.to(device)
optimizer_densenet = optim.Adam(model_densenet.parameters(), lr=0.0001)
#################################################################################################
"""
    Defining the model for Resnet18.
    Using Adam as an optimizer for the same
"""
model_resnet = models.resnet18(pretrained=True)
num_ftrs_resnet = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs_resnet, 3)
model_resnet.to(device)
optimizer_resnet = optim.Adam(model_resnet.parameters(), lr=0.0001)
#################################################################################################
"""
    Defining the model for GoogLeNet.
    Using Adam as an optimizer for the same
"""
model_googlenet = models.googlenet(pretrained=True)
num_ftrs_googlenet = model_googlenet.fc.in_features
model_googlenet.fc = nn.Linear(num_ftrs_googlenet, 3)
model_googlenet.to(device)
optimizer_googlenet = optim.Adam(model_googlenet.parameters(), lr=0.0001)
#################################################################################################
"""
    Helper function for prediction with a given 'model'
    and saving it to a prediction csv with the 'name' param.
"""


def makePrediction(model, name):
    model.eval()

    test_predictions = list()

    with torch.no_grad():
        for idx, batch in enumerate(dataloaders["test"]):
            data = batch["image"].to(device)
            result = model(data)
            pred = result.argmax(dim=1, keepdim=True)
            test_predictions += [p.item() for p in pred.flatten()]
            print("Test | Batch: {}/{}".format(idx + 1, len(dataloaders["test"])))

    test_predictions_df = pd.DataFrame(index=np.arange(len(test_predictions)))
    test_predictions_df.index.name = "ID"
    test_predictions_df["Label"] = test_predictions
    path = "./ckp/" + "predictions_" + name + ".csv"
    test_predictions_df.to_csv(path)


# Commented out IPython magic to ensure Python compatibility.
"""
    Helper function for making loss-accuracy graphs
    with a given set for 'train', 'val' and 'name'
"""


def makeGraphs(train_losses, val_losses, name):
    #     %matplotlib inline
    plt.figure(figsize=(12, 5))
    epoch_list = np.arange(1, 30 + 1)
    plt.xticks(epoch_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_list, train_losses, label="Training loss")
    plt.plot(epoch_list, val_losses, label="Validation loss")
    plt.legend(loc="upper right")
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("./ckp/" + name + ".png")
    print(train_losses, val_losses)


#################################################################################################
"""
    Dictionary for calling the loop on multiple models
    and training to output predictions csv for different
    models.
"""

models_list = {
    0: (model_densenet, optimizer_densenet, "DenseNet"),
    1: (model_resnet, optimizer_resnet, "Resnet"),
    2: (model_googlenet, optimizer_googlenet, "Googlenet"),
}
for i in models_list:
    model_f = models_list[i][0]
    optimizer_f = models_list[i][1]
    best, train_losses, val_losses = train_validate_model(
        model_f, device, dataloaders, optimizer_f, criterion, epochs=30
    )
    makeGraphs(train_losses, val_losses, models_list[i][2])
    makePrediction(best, models_list[i][2])
    try:
        os.remove("./ckp/best_model.pt")
    except:
        print("File not present, continuing")
#################################################################################################
"""
    Reading the predictions from three models and 
    performing voting to finalize the predictions.
"""
df1 = pd.read_csv("./ckp/predictions_Resnet.csv")
df2 = pd.read_csv("./ckp/predictions_Googlenet.csv")
df3 = pd.read_csv("./ckp/predictions_DenseNet.csv")
df1.columns = ["ID", "Resnet"]
df2.columns = ["ID", "GoogleNet"]
df3.columns = ["ID", "DenseNet"]
df = pd.concat([df1, df2.GoogleNet, df3.DenseNet], axis=1)
final_labels = np.zeros(len(df), dtype=np.int64)
for index, row in df.iterrows():
    temp = {0: 0, 1: 0, 2: 0}
    temp[row.Resnet] += 1
    temp[row.GoogleNet] += 1
    temp[row.DenseNet] += 1
    flag = 0
    for key, value in temp.items():
        if value >= 2:
            final_labels[index] = key
            flag = 1
    if flag == 0:
        final_labels[index] = row.Resnet
df["Label"] = final_labels
df.drop(["Resnet", "GoogleNet", "DenseNet"], axis=1, inplace=True)
df.to_csv("test_result.csv", index=False)
