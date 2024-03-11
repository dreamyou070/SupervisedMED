import os
from PIL import Image
def main() :

    base_dir = '/home/dreamyou070/MyData/anomal_source'
    save_base_dir = '/home/dreamyou070/MyData/anomal_source_L_mode'

    folders = os.listdir(base_dir)
    for folder in folders :
        folder_dir = os.path.join(base_dir, folder)
        save_folder_dir = os.path.join(save_base_dir, folder)
        os.makedirs(save_folder_dir, exist_ok = True)
        categories = os.listdir(folder_dir)
        for cat in categories :
            cat_dir = os.path.join(folder_dir, cat)
            save_cat_dir = os.path.join(save_folder_dir, cat)
            os.makedirs(save_cat_dir, exist_ok = True)
            images = os.listdir(cat_dir)
            for img in images :
                img_dir = os.path.join(cat_dir, img)
                print(img_dir)
                pil = Image.open(img_dir).convert('L')
                pil.save(os.path.join(save_cat_dir, img))

if __name__ == '__main__' :
    main()