import os
from PIL import Image, ImageFilter, ImageEnhance
import torch
from perlin import rand_perlin_2d_np
import numpy as np
import cv2

# [1] random img
resize_shape = 512
org_img_dir = '109.JPG'
org_pil = Image.open(org_img_dir).convert('RGB').resize((resize_shape,resize_shape))

# [2] change shadow img
# (1) blurring pseudo : pseudo_pil = org_pil.filter(ImageFilter.GaussianBlur(5))
# (2) pseudo_pil = ImageEnhance.Brightness(org_pil).enhance(0.7)
# (3) contast
pseudo_pil = ImageEnhance.Contrast(org_pil).enhance(2)



# [2] random mask
min_perlin_scale, max_perlin_scale = 2,6
perlin_scalex = 2 ** (torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0])
perlin_scaley = 2 ** (torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0])
perlin_noise = rand_perlin_2d_np((resize_shape, resize_shape), (perlin_scalex, perlin_scaley))
threshold = 0.3
perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
# smoothing
perlin_thr_ = cv2.GaussianBlur(perlin_thr, (3,3), 0)
perlin_thr = np.expand_dims(perlin_thr_, axis=2)
# only on object

org_np = np.array(org_pil) * (1-perlin_thr) + np.array(pseudo_pil) * (perlin_thr)
recon_pil = Image.fromarray(org_np.astype(np.uint8))
recon_pil.show()
Image.fromarray((perlin_thr_ * 255)).show()

