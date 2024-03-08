import os
from PIL import Image, ImageFilter, ImageEnhance
import torch
from perlin import rand_perlin_2d_np
import numpy as np
import cv2


img = np.ones((512,512,3))
new_np = np.zeros_like(img)
new_np[:, :, 0] = img[:,:,0]