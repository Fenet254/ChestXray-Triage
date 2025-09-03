import os
import subprocess
import zipfile

def download_datasets():
    # Create raw data directories
    os.makedirs("data/raw/pneumonia", exist_ok=True)
    os.makedirs("data/raw/tuberculosis", exist_ok=True)
    
    # Download Pneumonia dataset
    print("Downloading Pneumonia dataset...")
    os.chdir("data/raw/pneumonia")
    subprocess.run([
        "kaggle", "datasets", "download", 
        "-d", "paultimothymooney/chest-xray-pneumonia"
    ])
    
    # Extract and clean up
    with zipfile.ZipFile("chest-xray-pneumonia.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove("chest-xray-pneumonia.zip")
    os.chdir("../../../")
    
    # Download Tuberculosis dataset
    print("Downloading Tuberculosis dataset...")
    os.chdir("data/raw/tuberculosis")
    subprocess.run([
        "kaggle", "datasets", "download", 
        "-d", "usmansaeed/tuberculosis-tb-chest-xray-dataset"
    ])
    
    # Extract and clean up
    with zipfile.ZipFile("tuberculosis-tb-chest-xray-dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove("tuberculosis-tb-chest-xray-dataset.zip")
    os.chdir("../../../")
    
    print("Download complete!")

if __name__ == "__main__":
    download_datasets()