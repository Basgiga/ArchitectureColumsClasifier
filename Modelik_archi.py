import cv2 as cv
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# wczytrywanie
class CustomImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        labels_path = os.path.join(root_dir, "labels.npy")
        self.labels = np.load(labels_path)

        for filename in sorted(os.listdir(root_dir)):
            if filename.startswith("padded_") and filename.endswith(
                    ('.jpg', '.jpeg', '.png', '.jfif', '.webp')):
                img_path = os.path.join(root_dir, filename)
                self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

root_directory = r"D:\Pobrane_Opera\Computer_Vision\spaddowane_all"
def test_labeli():
    dataset = CustomImageDataset(root_directory, transform=None)

    for i in range(len(dataset)):
        image, label = dataset[i]
        image_np = np.array(image)

        plt.figure()
        plt.imshow(image_np)
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()

class Sobel(torch.nn.Module):
    def __init__(self, normalized=True):
        super(Sobel, self).__init__()
        self.normalized = normalized

        self.filter_x = torch.tensor([[-2, 0, 2],
                                      [-2, 0, 2],
                                      [-2, 0, 2]])

        self.filter_y = torch.tensor([[-2, -3, -2],
                                      [0, 0, 0],
                                      [2, 3, 2]])
    def forward(self, img):
        img = transforms.ToTensor()(img)

        img_x = F.conv2d(img.unsqueeze(0), self.filter_x.unsqueeze(0).unsqueeze(0).float(), padding=1)
        img_y = F.conv2d(img.unsqueeze(0), self.filter_y.unsqueeze(0).unsqueeze(0).float(), padding=1)

        grad_magnitude = torch.sqrt(img_x**2 + img_y**2)

        if self.normalized:
            grad_magnitude = grad_magnitude / grad_magnitude.max()

        return grad_magnitude[0]

transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.GaussianBlur(kernel_size=3),
    transforms.ColorJitter(contrast=0.5, brightness=0.8),
    Sobel(normalized=True),
    transforms.ColorJitter(contrast=0.5),
])

dataset = CustomImageDataset(root_directory, transform=transform)
print(len(dataset))
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def test_filtrow(dataloader, num_images=100):
    num_displayed = 0
    for images, labels in dataloader:
        for i in range(images.shape[0]):
            if num_displayed >= num_images:
                break

            image = images[i]
            label = labels[i]
            image_np = image.squeeze().numpy()

            plt.figure()
            plt.imshow(image_np, cmap='gray')
            plt.title(f"Label: {label}")
            plt.axis('off')
            plt.show()

            num_displayed += 1
        if num_displayed >= num_images:
            break

"""
print("Training dane:")
for batch_idx, (images, labels) in enumerate(train_dataloader):
    print(f"Batch {batch_idx}: Images shape {images.shape}, Labels: {labels}")

print("\nTest dane:")
for batch_idx, (images, labels) in enumerate(test_dataloader):
    print(f"Batch {batch_idx}: Images shape {images.shape}, Labels: {labels}")
"""

def siecCNN():
    from time import time

    class modelik(nn.Module):
        def __init__(self,
                     input_shapes: int,
                     hidden_units: int,
                     output_shapes: int, ):
            super().__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shapes,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units * 2,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(hidden_units * 2),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units * 2,
                          out_channels=hidden_units * 2,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(hidden_units * 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

            self.conv_block_3 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units * 2,
                          out_channels=hidden_units * 3,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(hidden_units * 3),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units * 3,
                          out_channels=hidden_units * 3,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(hidden_units * 3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_block_4 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units * 3,
                          out_channels=hidden_units * 4,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(hidden_units * 4),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units * 4,
                          out_channels=hidden_units * 4,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(hidden_units * 4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units * 4 * 32 * 32,
                          out_features=512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(in_features=512, out_features=output_shapes)
            )

        def forward(self, x):
            x = self.conv_block_1(x)
            #print(x.shape)
            x = self.conv_block_2(x)
            #print(x.shape)
            x = self.conv_block_3(x)
            #print(x.shape)
            x = self.conv_block_4(x)
            #print(x.shape)
            x = self.classifier(x)
            return x

    model = modelik(input_shapes=1,  # ile kolorow
                    hidden_units=40,
                    output_shapes=3).to(device)
    return (model)

def uczenie():
    model = siecCNN()
    # loss fn i optimizer
    weights = torch.tensor([1.5, 1.5, 1.0])  # zwiekszamy wage dla kariatyd (klasa 1)
    weights = weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00002)

    # train

    # start = time.time()
    epochs = 30

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-----")
        train_loss = 0
        for batch, (image, label) in enumerate(train_dataloader):
            model.train()
            image, label = image.to(device), label.to(device).long()

            # Forward pass
            y_pred = model(image)

            # loss
            loss = loss_fn(y_pred, label)
            train_loss += loss.item()

            # Back
            optimizer.zero_grad()
            loss.backward()

            # Optimizer
            optimizer.step()

            #if batch % 100 == 0:
                #print(f"Processed {batch * len(image)}/{len(train_dataloader.dataset)} samples")

        train_loss /= len(train_dataloader)

        # Ewaluacja
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for X_test, y_test in test_dataloader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                test_pred = model(X_test)
                test_loss += loss_fn(test_pred, y_test.long()).item()
                test_acc += (test_pred.argmax(dim=1) == y_test).sum().item() / len(y_test)

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(
            f"\nTrain loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}"
        )
        if(epoch >= 3 and test_acc >= 0.995):
            print("osiagnieto satystfakcjonujacy wynik!")
            break
    return(model)


MODEL_PATH = Path(r"D:\Pobrane_Opera\ctwitch\modele")
def zapismodelu(model_path,model):
    nazwa = "modelik_na_archi_159_11_24.pth"
    model_save_path = model_path / nazwa
    torch.save(obj = model.state_dict(),f = model_save_path)
    print("model zostal zapisany :))")



def predykcja(images_path1, model_save_path):
    model = siecCNN()
    model.load_state_dict(torch.load(f=model_save_path))
    model.to(device)


    pr_path_names = []
    for filename in os.listdir(images_path1):
        if filename.startswith("padded_") and filename.endswith(
                ('.jpg', '.jpeg', '.png', '.jfif', '.webp')):
            img_path1 = os.path.join(images_path1, filename)
            pr_path_names.append(img_path1)

    for image_paths in pr_path_names:
        image = Image.open(image_paths).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output).item()
        class_names = ['greckie', 'kariatydy', 'rzymskie']
        predicted_class_name = class_names[predicted_class]
        print(f"klasa dla obrazka {image_paths}: {predicted_class_name}")



#test_labeli()
#test_filtrow(train_dataloader)

modelik = uczenie()
zapismodelu(model_path=MODEL_PATH,model=modelik)

predykcja(images_path1="D:\Pobrane_Opera\ctwitch\spredykcja_padded",model_save_path="D:\Pobrane_Opera\ctwitch\modele\modelik_na_archi_159_11_24.pth")

# end = time.time()
# print(f"trening zajal: {end - start:.2f} sekund")