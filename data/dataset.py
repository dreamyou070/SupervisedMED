import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from torchvision import transforms
import cv2

def passing_mvtec_argument(args):
    global argument
    global anomal_p
    global do_rot_augment

    argument = args
    anomal_p = args.anomal_p
    do_rot_augment = args.do_rot_augment


class TestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample

class TrainDataset(Dataset):

    def __init__(self,
                 root_dir,
                 anomaly_source_path,
                 resize_shape=None,
                 tokenizer=None,
                 caption: str = None,
                 use_perlin: bool = False,
                 anomal_only_on_object: bool = True,
                 anomal_training: bool = False,
                 latent_res: int = 64,
                 do_anomal_sample: bool = True,
                 use_object_mask: bool = True):

        # [1] base image
        self.root_dir = root_dir
        image_paths, gt_paths = [], []
        folders = os.listdir(self.root_dir)
        for folder in folders:
            # folder = normal, anormal
            folder_dir = os.path.join(self.root_dir, folder)
            rgb_folder = os.path.join(folder_dir, 'rgb')
            gt_folder = os.path.join(folder_dir, 'gt')
            images = os.listdir(rgb_folder)
            for image in images:
                image_path = os.path.join(rgb_folder, image)
                image_paths.append(image_path)
                gt_paths.append(os.path.join(gt_folder, image))

        self.resize_shape = resize_shape
        if do_anomal_sample:
            assert anomaly_source_path is not None, "anomaly_source_path should be given"

        if anomaly_source_path is not None:
            self.anomaly_source_paths = []
            for ext in ["png", "jpg"]:
                self.anomaly_source_paths.extend(sorted(glob.glob(anomaly_source_path + f"/*/*/*.{ext}")))
        else:
            self.anomaly_source_paths = []

        self.caption = caption
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5]), ])
        self.use_perlin = use_perlin

        self.image_paths = image_paths
        self.gt_paths = gt_paths

        self.anomal_only_on_object = anomal_only_on_object
        self.anomal_training = anomal_training
        self.latent_res = latent_res

    def __len__(self):
        if len(self.anomaly_source_paths) > 0:
            return max(len(self.image_paths), len(self.anomaly_source_paths))
        else:
            return len(self.image_paths)

    def torch_to_pil(self, torch_img):
        # torch_img = [3, H, W], from -1 to 1
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil = Image.fromarray(np_img)

    def get_img_name(self, img_path):
        rgb_folder, name = os.path.split(img_path)
        net_name, ext = os.path.splitext(name)
        # class_folder, rgb = os.path.split(rgb_folder)
        # return name, class_folder
        return net_name

    def get_object_mask_dir(self, img_path):
        parent, name = os.path.split(img_path)
        parent, _ = os.path.split(parent)
        object_mask_dir = os.path.join(parent, f"object_mask/{name}")
        return object_mask_dir

    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask

    def load_image(self, image_path, trg_h, trg_w, type='RGB'):
        image = Image.open(image_path)
        if type == 'RGB':
            if not image.mode == "RGB":
                image = image.convert("RGB")
        elif type == 'L':
            if not image.mode == "L":
                image = image.convert("L")
        if trg_h and trg_w:
            image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(image, np.uint8)
        return img

    def __getitem__(self, idx):

        # [1] base
        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]
        img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1])  # np.array,

        # [2] gt dir
        gt_path = self.gt_paths[img_idx]
        gt_img = np.array(
            Image.open(gt_path).convert('L').resize((self.latent_res, self.latent_res), Image.BICUBIC))  # 64,64
        gt_torch = torch.tensor(gt_img) / 255
        gt_torch = torch.where(gt_torch > 0, 1, 0).unsqueeze(0)

        if self.tokenizer is not None:
            input_ids, attention_mask = self.get_input_ids(self.caption)  # input_ids = [77]
        else:
            input_ids = torch.tensor([0])

        return {'image': self.transform(img),  # [3,512,512]
                "gt": gt_torch,  # [1, 64, 64]
                'input_ids': input_ids.squeeze(0), }