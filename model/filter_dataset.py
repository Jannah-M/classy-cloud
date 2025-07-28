import os
import shutil

source_folder = "/Users/jannahmansoor/downloads/ccsn_v2" 

destination_folder = "data/sample_clouds"

# Changing labels
folders_to_copy = 
{
    "Cu": "cumulus",
    "Ci": "cirrus",
    "St": "stratus"
}

for folder_name in folders_to_copy:
    new_name = folders_to_copy[folder_name]
    
    source_path = os.path.join(source_folder, folder_name)
    destination_path = os.path.join(destination_folder, new_name)

    os.makedirs(destination_path, exist_ok=True)

    for filename in os.listdir(source_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            src = os.path.join(source_path, filename)
            dst = os.path.join(destination_path, filename)
            shutil.copyfile(src, dst)

print("Done copying the three cloud types.")
