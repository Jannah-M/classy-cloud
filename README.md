# Classy Cloud

ClassyCloud is a simple web app that classifies cloud images into Cirrus, Cumulus, or Stratus using a PyTorch CNN model.  
The app is built with Streamlit for the interface and deployed to [Streamlit Cloud](https://classycloud.streamlit.app).


## Features
- Ability to instantly predict cloud type based on an uploaded image.
- Trained PyTorch CNN model (`cloud_cnn.pth`) for three cloud categories.  
- Fast and lightweight web interface powered by Streamlit, with hand-draw illustrations by me!


## Tech Stack
- Python 3.10
- PyTorch for the convolutional neural network.
- Torchvision for image transforms.
- Streamlit for the interactive UI
- JS Paint for illustrations


## Demo
Try it live: [classycloud.streamlit.app](https://classycloud.streamlit.app)

Please note that this project is hosted using the free tier of Streamlit. This means that after a period of inactivity (usually around one day), the app will need to be reawakened by clicking the "Wake Up" button prompted by Streamlit. This restarts the app, and will take about 30 seconds to load. 

## Local Setup

Clone the repo and run locally:

```bash
git clone https://github.com/Jannah-M/classy-cloud.git
cd classy-cloud
pip install -r requirements.txt
streamlit run gui/app.py
