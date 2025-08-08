import os
import random
import shutil  

source_dir = "data/sample_clouds"

dest_dir = "data/sample_clouds_split"

train_ratio = 0.8 # 80/20 train/test

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
    
os.makedirs(dest_dir, exist_ok=True)


for folder in os.listdir(source_dir):
    if folder.startswith('.'):  # skips hidden files
        continue

    test_dest = f"{dest_dir}/test/{folder}"
    train_dest = f"{dest_dir}/train/{folder}"

    split_index = int(len(os.listdir(f"data/sample_clouds/{folder}")) * train_ratio)
    os.makedirs(test_dest, exist_ok=True)
    os.makedirs(train_dest, exist_ok=True)

    images = os.listdir(f"data/sample_clouds/{folder}")
    random.shuffle(images)

    for i in range(split_index):
        currentPic = images[i]
        shutil.copyfile(f"data/sample_clouds/{folder}/{currentPic}", f"{train_dest}/{currentPic}")
    

    for j in range(split_index, len(images)):
        currentPic = images[j]
        shutil.copyfile(f"data/sample_clouds/{folder}/{currentPic}", f"{test_dest}/{currentPic}")

print("Data has been split into train and test sets.")
