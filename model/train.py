import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
])


train_dir = "data/sample_clouds_split/train"
test_dir = "data/sample_clouds_split/test"

train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


class CloudCNN(nn.Module):
    def __init__(self):
        super(CloudCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        self.fc1 = nn.Linear(28800, 64)   
        self.fc2 = nn.Linear(64, 3)     # 3 classes: cumulus, cirrus, stratus

    def forward(self, x):
        
        # Convolution then ReLU then Pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # 2nd Convolution then ReLU then Pooling
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

model = CloudCNN()

num_epochs = 20 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

avg = 0
for epoch in range(num_epochs):
    avg = 0  # reset average loss for this epoch
    for images, labels in train_loader:
        optimizer.zero_grad()       # clear gradients for this batch
        outputs = model(images)     # forward pass
        loss = criterion(outputs, labels)  # calculate loss
        loss.backward()             # backpropagate
        optimizer.step()            # update weights
        avg += loss.item()          # update loss

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg / len(train_loader):.4f}") # print the epoch # and average loss

torch.save(model.state_dict(), "cloud_cnn.pth")

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)              # forward pass
        _, predicted = torch.max(outputs, 1) # get class with highest score
        total += labels.size(0)              # count samples in batch
        correct += (predicted == labels).sum().item()  # count correct ones

print(f"Test Accuracy: {100 * correct / total:.2f}%")
