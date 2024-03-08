import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from PIL import Image
import numpy as np
import os
import random

def main() :

    # [0]
    save_base_dir = 'noise_source'
    os.makedirs(save_base_dir, exist_ok = True)
    """
    # [1] simple perlin
    save1 = os.path.join(save_base_dir, 'simple_noise')
    os.makedirs(save1, exist_ok = True)
    repeat_time = 50
    for i in range(repeat_time) :
        rand_i = random.randint(3, 20)
        noise = PerlinNoise(octaves=rand_i, seed=1)
        xpix, ypix = 100, 100
        pic = [[noise([i / xpix, j / ypix]) for j in range(xpix)] for i in range(ypix)]
        plt.imshow(pic, cmap='gray')
        plt.savefig(os.path.join(save1, f'noise_{i}.jpg'))
    """

    # [2] mixed perlin
    save2 = os.path.join(save_base_dir, 'mixed_noise')
    os.makedirs(save2, exist_ok=True)
    repeat_time = 50
    for ii in range(repeat_time):

        noise1 = PerlinNoise(octaves=random.randint(3, 20))
        noise2 = PerlinNoise(octaves=random.randint(3, 20))
        noise3 = PerlinNoise(octaves=random.randint(3, 20))
        noise4 = PerlinNoise(octaves=random.randint(3, 20))

        xpix, ypix = 100, 100
        pic = []
        for i in range(xpix):
            row = []
            for j in range(ypix):
                noise_val = noise1([i / xpix, j / ypix])
                noise_val += 0.5 * noise2([i / xpix, j / ypix])
                noise_val += 0.25 * noise3([i / xpix, j / ypix])
                noise_val += 0.125 * noise4([i / xpix, j / ypix])

                row.append(noise_val)
            pic.append(row)
        plt.imshow(pic, cmap='gray')
        plt.savefig(os.path.join(save2, f'multi_noise_{ii}.jpg'))




if __name__ == '__main__' :
    main()
