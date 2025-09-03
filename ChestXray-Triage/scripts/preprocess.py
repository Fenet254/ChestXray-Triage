import os
import shutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():
    print("Starting data preprocessing...")
    
    # Create processed directory structure
    base_dir = "data/processed"
    for folder in ['train', 'val', 'test']:
        for category in ['pneumonia', 'tuberculosis', 'normal']:
            os.makedirs(f"{base_dir}/{folder}/{category}", exist_ok=True)
    
    # Process datasets
    process_pneumonia_data()
    process_tuberculosis_data()
    
    # Create metadata CSV
    create_metadata_csv()
    
    print("Data preprocessing complete!")

def process_pneumonia_data():
    print("Processing Pneumonia dataset...")
    
    # Path to pneumonia data
    pneumonia_path = "data/raw/pneumonia/chest_xray"
    
    # Get all pneumonia images
    pneumonia_images = []
    for root, dirs, files in os.walk(os.path.join(pneumonia_path, "train", "PNEUMONIA")):
        for file in files:
            if file.endswith(('.jpeg', '.jpg', '.png')):
                pneumonia_images.append(os.path.join(root, file))
    
    # Get normal images from pneumonia dataset
    normal_images = []
    for root, dirs, files in os.walk(os.path.join(pneumonia_path, "train", "NORMAL")):
        for file in files:
            if file.endswith(('.jpeg', '.jpg', '.png')):
                normal_images.append(os.path.join(root, file))
    
    # Sample 500 pneumonia and 200 normal images
    sampled_pneumonia = random.sample(pneumonia_images, min(500, len(pneumonia_images)))
    sampled_normal = random.sample(normal_images, min(200, len(normal_images)))
    
    # Split into train/val/test (70/15/15)
    pneumonia_train, pneumonia_temp = train_test_split(sampled_pneumonia, test_size=0.3, random_state=42)
    pneumonia_val, pneumonia_test = train_test_split(pneumonia_temp, test_size=0.5, random_state=42)
    
    normal_train, normal_temp = train_test_split(sampled_normal, test_size=0.3, random_state=42)
    normal_val, normal_test = train_test_split(normal_temp, test_size=0.5, random_state=42)
    
    # Copy images to processed directory
    copy_images(pneumonia_train, "pneumonia", "train")
    copy_images(pneumonia_val, "pneumonia", "val")
    copy_images(pneumonia_test, "pneumonia", "test")
    
    copy_images(normal_train, "normal", "train")
    copy_images(normal_val, "normal", "val")
    copy_images(normal_test, "normal", "test")

def process_tuberculosis_data():
    print("Processing Tuberculosis dataset...")
    
    # Path to tuberculosis data
    tb_path = "data/raw/tuberculosis/TB_Chest_Radiography_Database"
    
    # Get all tuberculosis images
    tb_images = []
    for root, dirs, files in os.walk(os.path.join(tb_path, "Tuberculosis")):
        for file in files:
            if file.endswith(('.jpeg', '.jpg', '.png')):
                tb_images.append(os.path.join(root, file))
    
    # Get normal images from tuberculosis dataset
    normal_images = []
    for root, dirs, files in os.walk(os.path.join(tb_path, "Normal")):
        for file in files:
            if file.endswith(('.jpeg', '.jpg', '.png')):
                normal_images.append(os.path.join(root, file))
    
    # Sample 500 TB images
    sampled_tb = random.sample(tb_images, min(500, len(tb_images)))
    
    # Add to our normal images if we need more
    if len(normal_images) > 0:
        # We already have normal images from pneumonia, so we'll just use these for TB
        pass
    
    # Split into train/val/test (70/15/15)
    tb_train, tb_temp = train_test_split(sampled_tb, test_size=0.3, random_state=42)
    tb_val, tb_test = train_test_split(tb_temp, test_size=0.5, random_state=42)
    
    # Copy images to processed directory
    copy_images(tb_train, "tuberculosis", "train")
    copy_images(tb_val, "tuberculosis", "val")
    copy_images(tb_test, "tuberculosis", "test")

def copy_images(image_paths, category, split):
    """Copy images to the processed directory with new names"""
    for i, src_path in enumerate(image_paths):
        # Create new filename
        filename = f"{category}_{split}_{i:04d}{os.path.splitext(src_path)[1]}"
        dst_path = os.path.join("data/processed", split, category, filename)
        
        # Copy the image
        shutil.copy2(src_path, dst_path)

def create_metadata_csv():
    print("Creating metadata CSV...")
    
    metadata = []
    base_dir = "data/processed"
    
    # Walk through all processed images
    for split in ['train', 'val', 'test']:
        for category in ['pneumonia', 'tuberculosis', 'normal']:
            category_path = os.path.join(base_dir, split, category)
            
            for filename in os.listdir(category_path):
                if filename.endswith(('.jpeg', '.jpg', '.png')):
                    filepath = os.path.join(category_path, filename)
                    
                    # Create metadata entry
                    metadata.append({
                        'image_id': filename,
                        'filepath': filepath,
                        'pneumonia': 1 if category == 'pneumonia' else 0,
                        'tb': 1 if category == 'tuberculosis' else 0,
                        'normal': 1 if category == 'normal' else 0,
                        'source': 'pneumonia_dataset' if category in ['pneumonia', 'normal'] else 'tb_dataset',
                        'split': split
                    })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(metadata)
    df.to_csv("data/processed/metadata.csv", index=False)
    print(f"Metadata saved with {len(df)} entries")

if __name__ == "__main__":
    preprocess_data()