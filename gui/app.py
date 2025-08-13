import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import transforms
from PIL import Image

import streamlit as st
import numpy as np
import pandas as pd


class CloudCNN(nn.Module):
    def __init__(self):
        super(CloudCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        self.fc1 = nn.Linear(28800, 64)   # fill in ??? later
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


model = CloudCNN() # If you saved the state_dict
model.load_state_dict(torch.load("cloud_cnn.pth", map_location=torch.device("cpu")))
model.eval() # Set model to evaluation mode

st.title('Classy Cloud\'s Cloud Classifier ⛅️')


left_column, middle_column, right_column = st.columns(3)
# You can use a column just like st.sidebar:

with middle_column:
    image = Image.open('gui/classyTitle.png') # Replace 'my_image.jpg' with your image file path
st.image(image, width=500)

st.write(f"Upload an image of a cloud, and let Classy do the thinking! She'll tell you what kind of cloud formation is in the picture!")


uploaded_file = st.file_uploader("")
if uploaded_file is not None:
    # To read file as bytes:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Cloud Pic", use_container_width=True)
    

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    class_names = ["Cirrus", "Cumulus", "Stratus"]

    st.success(f"Your cloud is a {class_names[predicted.item()]}!")

