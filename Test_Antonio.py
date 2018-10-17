# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:03:49 2018

@author: Antonio
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import io

img = io.imread('Images/face_bouchon.jpg', as_gray=True)

s = np.linspace(0, 2*np.pi, 400)
x = 2583 + 100*np.cos(s)
y = 2551 + 100*np.sin(s)
init = np.array([x, y]).T

snake = active_contour(gaussian(img, 3),
                       init, alpha=0.015, beta=10, gamma=0.001)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])












#
#if __name__ == '__main__':
#
#    
#    edge_sobel = sobel(img)

    #calculeHauteur(edge_sobel, 0.1, True)
 #   trouverSeuilContrasteOptimal(edge_sobel, 112)