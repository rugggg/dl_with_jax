import numpy as np
from scipy.signal import convolve2d
from matplotlib import figure, pyplot as plt
from skimage.io import imread, imsave
from skimage.util import img_as_float32, img_as_ubyte, random_noise


img = imread('grouch.jpg')

plt.figure(figsize= (6,10))
plt.imshow(img)
# plt.show()
print(img.shape)

oscar_face = img[75:290, 150:390]
plt.imshow(oscar_face)
# plt.show()

# flip oscar
plt.imshow(oscar_face[:, ::-1, :])
plt.show()

img_noised = random_noise(img, mode='gaussian')

# using finite impulse response filters - one of which is a basic kernel filter
# or a convolution

# simple blur
# makes a matrix of ones
kernel_blur = np.ones((5,5))
# then assigns equal weight
kernel_blur /= np.sum(kernel_blur)

def gaussian_kernel(kernel_size, sigma=1.0, mu=0.0):
    center = kernel_size // 2
    x, y = np.mgrid[
            -center : kernel_size - center,
            - center : kernel_size - center]
    d = np.sqrt(np.square(x) + np.square(y))
    koeff = 1/(2 * np.pi * np.square(sigma))
    kernel = koeff * np.exp(-np.square(d-mu)/(2 * np.square(sigma)))
    return kernel

kernel_gauss = gaussian_kernel(5)
print(kernel_gauss)


def color_convolution(image, kernel):
    channels = []
    for i in range(image.shape[-1]):
        color_channel = image[:,:,i]
        filtered_channel = convolve2d(color_channel, kernel, mode='same')
        filtered_channel = np.clip(filtered_channel, 0.0, 1.0)
        channels.append(filtered_channel)
    final_image = np.stack(channels, axis=2)
    return final_image


img_blur = color_convolution(img_noised, kernel_gauss)
plt.figure(figsize = (12, 10))
plt.imshow(np.hstack((img_noised, img_blur)))
plt.show()
