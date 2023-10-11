# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 23:27:52 2023

@author: soren
"""

import numpy as np
from PIL import Image
import scipy as sp
import copy
from scipy import ndimage
import skimage.metrics

#import line mask
line_path = Image.open("C:/Users/soren/OneDrive/Documents/Physics Bachelors/Year 3/Computational Physics - FYTN03/Module B/Masks/masking_image3.png")
line_path = line_path.convert("L")
line_image = np.array(line_path)
#import text mask
masking_path = Image.open("C:/Users/soren/OneDrive/Documents/Physics Bachelors/Year 3/Computational Physics - FYTN03/Module B/Masks/Mask 2.png")
masking_path = masking_path.convert("L")
text_image = np.array(masking_path)
#import example photo
glacier = Image.open("C:/Users/soren/OneDrive/Documents/Physics Bachelors/Year 3/Computational Physics - FYTN03/Module B/Trial photos/Glacier.png")
glacier = glacier.convert("L")  #convert to Grayscale
glacier = np.array(glacier)
#import another
mountain = Image.open("C:/Users/soren/OneDrive/Documents/Physics Bachelors/Year 3/Computational Physics - FYTN03/Module B/Trial photos/test.og.png")
mountain = mountain.convert("L")  #convert to Grayscale
mountain = np.array(mountain)

line = np.where(line_image<252, 0, 1)
text = np.where(text_image<252, 0, 1)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

def c2(gradnorm, kappa, D):  #Perona Malik function
    return D/(1+(gradnorm/kappa)**2)

def time_varying_dt(dt, iteration):  #use a time-varying dt for better results
    return 1.e-10 + dt/(2+np.log(iteration))

#input both mask and original image as arrays
def anisotropic(mask, original, dt, kappa, D, iterations):
    pattern = np.where(mask<0.5, 1, 0)
    corrupted = original*mask
    corrupted, original = corrupted.astype(np.float64), original.astype(np.float64)
    reference = copy.deepcopy(original)  #make copy for loss calculation 
    error2 = skimage.metrics.structural_similarity(reference, corrupted)
    iteration = 0
    while error2<0.959:
        h = time_varying_dt(dt, iteration)
        Ix = ndimage.convolve(corrupted, sobel_x) #use kernel to approximate local gradients
        Iy = ndimage.convolve(corrupted, sobel_y)
        G = np.sqrt(Ix**2+Iy**2)
        diffusivity = c2(G, kappa, D)
        Ixx = ndimage.convolve(diffusivity*Ix, sobel_x)
        Iyy = ndimage.convolve(diffusivity*Iy, sobel_y)
        corrupted += h*(Ixx + Iyy)*pattern
        error2 = skimage.metrics.structural_similarity(reference, corrupted)
        print(error2)
        iteration += 1
        if iteration == iterations:
            print('Max iterations reached')
            corrupted = np.clip(corrupted, 0, 255)  #clip all values between 0 and 255 for image
            restored = corrupted.astype(np.int32)
            return restored, error2
    corrupted = np.clip(corrupted, 0, 255)
    restored = corrupted.astype(np.int32)
    return restored, error2

dt = 0.5 #0.15
D = 0.19
kappa = 50

result, loss = anisotropic(line, mountain, dt, kappa, D, 2000)
result = Image.fromarray((result).astype('uint8'))
result.show()
print(loss) #0.97
