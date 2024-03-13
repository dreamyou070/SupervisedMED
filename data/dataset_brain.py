import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
from PIL import Image, ImageFilter
from torchvision import transforms
import cv2
from data.perlin import rand_perlin_2d_np

anomal_p = 0.03

def passing_mvtec_argument(args):
    global argument

    argument = args


class TrainDataset_Brain(Dataset):

    def __init__(self,
                 root_dir,
                 anomaly_source_path=None,
                 anomal_position_source_path = None,
                 resize_shape=None,
                 tokenizer=None,
                 caption: str = None,
                 latent_res: int = 64):

        # [1] base image
        self.root_dir = root_dir
        image_paths, gt_paths, object_masks = [], [], []
        folders = os.listdir(self.root_dir)
        for folder in folders:
            repeat, cat = folder.split('_')
            folder_dir = os.path.join(self.root_dir, folder)
            rgb_folder = os.path.join(folder_dir, 'xray')
            images = os.listdir(rgb_folder)
            for image in images:
                for _ in range(int(repeat)):
                    image_path = os.path.join(rgb_folder, image)
                    image_paths.append(image_path)

        self.resize_shape = resize_shape
        self.caption = caption
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5]), ])
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.object_masks = object_masks
        self.latent_res = latent_res

        if anomaly_source_path is not None:
            self.anomaly_source_paths = []
            for ext in ["png", "jpg"]:
                self.anomaly_source_paths.extend(sorted(glob.glob(anomaly_source_path + f"/*/*/*.{ext}")))
        else:
            self.anomaly_source_paths = []

        self.anomal_position_source_paths = []
        if anomal_position_source_path is not None :             
            files = os.listdir(anomal_position_source_path)
            self.anomal_position_source_paths = [os.path.join(anomal_position_source_path, file)for file in files]
                
    def __len__(self):

        if len(self.anomaly_source_paths) != 0:
            return max(len(self.image_paths), len(self.anomaly_source_paths))
        else:
            return len(self.image_paths)

    def torch_to_pil(self, torch_img):
        # torch_img = [3, H, W], from -1 to 1
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil = Image.fromarray(np_img)

    def load_image(self, image_path, trg_h, trg_w, type='RGB'):
        image = Image.open(image_path)
        if type == 'RGB' :
            if not image.mode == "RGB":
                image = image.convert("RGB")
        elif type == 'L':
            if not image.mode == "L":
                image = image.convert("L")
        if trg_h and trg_w:
            image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(image, np.uint8)
        return img

    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask

    def __getitem__(self, idx):

        is_ok = 1

        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]

        # [1.1] anomal pure position
        anomal_position_idx = idx % len(self.anomal_position_source_paths)
        anomal_position_file_path = self.anomal_position_source_paths[anomal_position_idx]
        mask = Image.open(anomal_position_file_path).convert('L')
        mask_blur = mask.filter(ImageFilter.GaussianBlur(3)).resize((self.resize_shape[0], self.resize_shape[1]))
        mask_blur = np.array(mask_blur) / 255

        # [1.2] object mask
        xray, name = os.path.split(img_path)
        parent, _ = os.path.split(xray)
        object_path = os.path.join(parent, f'object_mask/{name}')
        object_mask = Image.open(object_path).convert('L').resize((self.resize_shape[0], self.resize_shape[1]))
        object_mask = np.array(object_mask) / 255

        # [1.3] anomal position
        anomal_mask = mask_blur * object_mask

        # [2] origin image
        origin_np = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1], type='L')  # np.array,
        img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1], type='RGB')  # np.array,

        # [3] anomal source
        anomal_src_idx = idx % len(self.anomaly_source_paths)
        anomal_dir = self.anomaly_source_paths[anomal_src_idx]
        anomal_source_np = np.array(Image.open(anomal_dir).convert('L').resize((self.resize_shape[0], self.resize_shape[1])))
        
        # [4] anomal img
        anomal_np = origin_np * (1 - anomal_mask) + (anomal_mask) * anomal_source_np
        anomal_pil = Image.fromarray(anomal_np.astype(np.uint8)).convert('RGB')
        anomal_np = np.array(anomal_pil)
        
        # [5] 
        gt_pil = Image.fromarray((anomal_mask * 255).astype(np.uint8)).resize((self.latent_res, self.latent_res))
        gt_np = np.array(gt_pil) / 255
        gt_torch = torch.tensor(gt_np)
        gt_torch = torch.where(gt_torch>0.5, 1, 0)

        
        if self.tokenizer is not None:
            input_ids, attention_mask = self.get_input_ids(self.caption)  # input_ids = [77]
        else:
            input_ids = torch.tensor([0])

        return {'image': self.transform(img),  # [3,512,512]
                "gt": gt_torch * 0.0,  # [1, 64, 64]
                'input_ids': input_ids.squeeze(0),
                'is_ok': is_ok,
                'augment_img': self.transform(anomal_np),
                'augment_mask': gt_torch}