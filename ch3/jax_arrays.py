import numpy as np
from scipy.signal import convolve2d
from matplotlib import figure, pyplot as plt
from skimage.io import imread, imsave
from skimage.util import img_as_float32, img_as_ubyte, random_noise


img = imread('grouch.jpg')

plt.figure(figsize= (6,10))
plt.imshow(img)
plt.show()
