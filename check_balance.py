import os

normal_path = r"C:\Users\melak\Downloads\Chest_Xray_Final\train\NORMAL"
pneumonia_path = r"C:\Users\melak\Downloads\Chest_Xray_Final\train\PNEUMONIA"

normal_count = len(os.listdir(normal_path))
pneumonia_count = len(os.listdir(pneumonia_path))

print(f"NORMAL: {normal_count}, PNEUMONIA: {pneumonia_count}")
print("Model input shape:", model.input_shape)