import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist

data_PATH = '/Users/wdlee/Work/Source/fft/data/PACS_val'
paint = 1
dog = 1
catoon = 0

if paint:
  data_PATH = data_PATH+'/art_painting'
  print(data_PATH)
  if dog:
    data_PATH = data_PATH+'/dog'
# elif catoon: 
#   data_PATH = data_PATH+'/catoon'
#   if dog:
#       data_PATH = data_PATH+'/dog'
dog_paint = imread(data_PATH+'/pic_219.jpg')

print(dog_paint)
dark_image_grey = rgb2gray(dog_paint)
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(dog_paint)


dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap = 'gray');



def fourier_masker_ver(image):
    f_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(rgb2gray(image)))
    dark_image_grey_fourier[:75, 125:240] = 0
    dark_image_grey_fourier[-75:,125:240] = 0
    fig, ax = plt.subplots(1,3,figsize=(15,15))
    ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    ax[0].set_title('Masked Fourier', fontsize = f_size)
    ax[1].imshow(rgb2gray(image), cmap = 'gray')
    ax[1].set_title('Greyscale Image', fontsize = f_size);
    ax[2].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)), 
                     cmap='gray')
    ax[2].set_title('Transformed Greyscale Image', 
                     fontsize = f_size);
    
fourier_masker_ver(dog_paint)


plt.show()

# elephant_paint = imread(data_PATH+'/art_painting/elephant/pic_078.jpg')
# giraffe_paint = imread(data_PATH+'/art_painting//pic_078.jpg')
