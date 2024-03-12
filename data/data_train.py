import os

base_folder = 'home/dreamyou070/MyData/anomaly_detection/medical/brain/NFBS/train'


nefb_dataset_folder = os.path.join(base_folder, 'nefb_dataset_1')
normal_folder = os.path.join(base_fodler, '1_normal')
os.makedirs(normal_folder, exist_ok = True)
normal_xray_folder = os.path.join(normal_folder, 'xray')
os.makedirs(normal_xray_folder, exist_ok = True)
normal_object_mask_folder = os.path.join(normal_folder, 'object_mask')
os.makedirs(normal_object_mask_folder, exist_ok = True)

ids = os.listdir(nefb_dataset_folder)
for id in ids :
    id_path = os.path.join(nefb_dataset_folder, id)
    xray_folder = os.path.join(id_path, 'xray')
    object_mask_folder = os.path.join(id_path, 'object_mask')
    imgs = os.listdir(xray_folder)
    for img in imgs :
        org_img_path = os.path.join(xray_folder, img)
        org_mask_path = os.path.join(object_mask_folder, img)
        new_img_path = os.path.join(normal_xray_folder, img)
        new_mask_path = os.path.join(normal_object_mask_folder, img)
        os.rename(org_img_path, new_img_path)
        os.rename(org_mask_path, new_mask_path)