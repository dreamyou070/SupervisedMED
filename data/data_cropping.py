import os
from PIL import Image

base_folder = 'noise_source'
folders = os.listdir(base_folder)
for folder in folders :
    folder_dir = os.path.join(base_folder, folder)
    images = os.listdir(folder_dir)
    for img in images :
        img_path = os.path.join(folder_dir, img)
        pil = Image.open(img_path).convert('L')
        new_pil = pil.crop((150,60,510,420))
        new_pil.save(img_path)