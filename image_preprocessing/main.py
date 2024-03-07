import os
from PIL import Image
import numpy as np

crop_normal_folder = 'cropped/normal'
os.makedirs(crop_normal_folder, exist_ok = True)
crop_anormal_folder = 'cropped/anormal'
os.makedirs(crop_anormal_folder, exist_ok = True)

gt_folder = 'gt'
rgb_folder = 'rgb'
maxillomandibular_folder = 'maxillomandibular'
imgs = os.listdir(rgb_folder)
for img in imgs :
    gt_path = os.path.join(gt_folder, img)
    rgb_path = os.path.join(rgb_folder, img)
    mouth_path = os.path.join(maxillomandibular_folder, img)
    np_img = np.array(Image.open(mouth_path).convert('L'))
    h,w = np_img.shape
    h_list, w_list = [], []
    for h_idx in range(h) :
        for w_idx in range(w) :
            if np_img[h_idx, w_idx] != 0 :
                h_list.append(h_idx)
                w_list.append(w_idx)
    h0,hz = min(h_list), max(h_list)
    w0, wz = min(w_list), max(w_list)

    h0 = h0 - 30 if h0 > 30 else 0
    hz = hz + 30 if hz < h - 30 else h
    w0 = w0 - 30 if w0 > 30 else 0
    wz = wz + 30 if wz < w - 30 else w

    org_pil = Image.open(rgb_path).convert('L')
    new_pil = org_pil.crop((w0,h0,wz,hz))

    mouth_pil = Image.open(mouth_path).convert('L')
    new_mouth_pil = mouth_pil.crop((w0,h0,wz,hz))

    # [1] Contrast Histogram
    np_img = np.array(new_pil)
    hist, bins = np.histogram(np_img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)

    # History Equalization 공식
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # Mask처리를 했던 부분을 다시 0으로 변환
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[np_img]
    min_value = img2.min()
    img2 = img2 - min_value
    img2 = img2 / img2.max()
    img2 = (img2 * 255).astype(np.uint8)
    new_pil = Image.fromarray(img2)




    gt_pil = Image.open(gt_path).convert('L')
    new_gt_pil = gt_pil.crop((w0, h0, wz, hz))
    new_gt_np = np.array(new_gt_pil)
    normality = new_gt_np.sum()
    if normality == 0 :
        save_base_folder = crop_normal_folder
    else :
        save_base_folder = crop_anormal_folder
    save_rgb_folder = os.path.join(save_base_folder, 'rgb')
    os.makedirs(save_rgb_folder, exist_ok = True)
    save_gt_folder = os.path.join(save_base_folder, 'gt')
    os.makedirs(save_gt_folder, exist_ok=True)
    save_mouth_folder = os.path.join(save_base_folder, 'mouth')
    os.makedirs(save_mouth_folder, exist_ok=True)

    new_pil.save(os.path.join(save_rgb_folder, img))
    new_gt_pil.save(os.path.join(save_gt_folder, img))
    new_mouth_pil.save(os.path.join(save_mouth_folder, img))