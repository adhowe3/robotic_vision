import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MEAN = 0.5
STD = 0.5

class OysterClassifier(nn.Module):
    def __init__(self):
        super(OysterClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7,7))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3136, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.adaptive_pool(x)   # will make (Batch, 64,7,7)
        x = torch.flatten(x, 1)     # 64*7*7 = 3136
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# train function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# test function 
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def save_dataset_sample(dataloader, filename="oyster_sample.png"):
    # 1. Grab a single batch of images
    images, labels = next(iter(dataloader))
    
    # 2. Un-normalize the images (Vectorized for Grayscale)
    # We use the same MEAN and STD you used in your transform (e.g., 0.5)
    MEAN, STD = 0.5, 0.5 
    
    # Apply math to the whole batch at once: (Pixel * STD) + MEAN
    sample_imgs = (images.clone() * STD) + MEAN

    # 3. Create a grid and save it
    # We use 'normalize=False' because we just did it manually
    grid = make_grid(sample_imgs, nrow=8) 
    
    save_image(grid, filename)
    print(f"Saved a sample grid to {filename}")


def create_confusion_matrix(model, device, test_loader, class_names, file_name):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    
    # 1. Start with a slightly larger figure to prevent cramping
    fig, ax = plt.subplots(figsize=(12, 10)) 
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # 2. Plot without the internal labels first
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax, values_format='d')
    
    # 3. Explicitly set the Axis Labels back
    ax.set_xlabel('Predicted Species', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Species', fontsize=12, fontweight='bold')
    
    # 4. Shrink the tick labels (species names)
    ax.tick_params(axis="both", which="major", labelsize=9)
    
    # 5. Shrink the numbers inside the boxes if they are too big
    for text in disp.text_.ravel():
        text.set_fontsize(8)

    plt.title(f"Fish Species Confusion Matrix\n{file_name}", pad=20)

    # 6. MANUALLY adjust margins so labels aren't cut off
    # bottom=0.2 means give 20% of the space to the bottom labels
    # left=0.2 means give 20% to the left labels
    plt.subplots_adjust(bottom=0.2, left=0.2) 
    
    plt.savefig(file_name, dpi=300)
    print(f"Confusion matrix saved as {file_name}")
    plt.show()


if __name__ == "__main__":
    IS_TRAIN_MODE = True   # toggle for train or load saved model
    MODEL_PATH = "oyster_classifier.pt"

    ################## handle data loading and modification #####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OysterClassifier().to(device)
    im_size = 128
    train_transform = transforms.Compose([
        transforms.Resize((im_size,im_size)), # MAYBE TRY MORE RECTANGLE SHAPE?
        transforms.Grayscale(num_output_channels=1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[MEAN], std=[STD]) # Standard ImageNet stats
    ])

    # Use the same normalization for testing
    test_transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[MEAN], std=[STD])
    ])

    # create the datsets with ImageFolder, automatically does the folder classification for us
    train_dataset = datasets.ImageFolder(root="oyster_shell/train", transform=train_transform)
    test_dataset = datasets.ImageFolder(root='oyster_shell/test', transform=test_transform)

    # load the datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    #show the images
    save_dataset_sample(train_loader, "check_oyster.png")

    if IS_TRAIN_MODE==True:

        ############################ implement training loop #############################
        print("Starting training -- using: ", device)
        batch_size = 32
        epochs = 100
        lr = 0.001

        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, epochs+1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
        
        # save the model weights
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model weights saved to {MODEL_PATH}")

    ############################## Load the model for analysis ############################
    elif IS_TRAIN_MODE==False:
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Loaded weights from {MODEL_PATH}")
        except FileNotFoundError:
            print("Model file not found. please train model first")
            exit()

        create_confusion_matrix(model, device, test_loader, test_dataset.classes, "test_confusion_matrix.png")        