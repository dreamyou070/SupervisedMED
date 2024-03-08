import torch
from PIL import ImageEnhance, Image
import numpy as np

pil_path = '1.JPG'
pil = Image.open(pil_path).convert('L')

white_np = np.ones((512,512)) * 255
Image.fromarray(white_np.astype(np.uint8)).show()