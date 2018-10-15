#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:41:36 2018

@author: jb
"""

from skimage import io

im = io.imread("Images/IMG_20181015_104405.jpg", as_gray = True)

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature
sigma = 0.6

from skimage.filters import roberts, sobel, scharr, prewitt

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im, sigma = sigma, low_threshold = 0.00, high_threshold = 0.0)
edges2 = feature.canny(im, sigma = sigma, low_threshold = 0.1, high_threshold = 0.15)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=2$', fontsize=20)


plt.show()


edge_roberts = roberts(im)
edge_sobel = sobel(im)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                       figsize=(8, 4))

ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
ax[0].set_title('Roberts Edge Detection')

ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
ax[1].set_title('Sobel Edge Detection')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

from skimage.filters import gaussian
from skimage.segmentation import active_contour


img = im

s = np.linspace(0, 2*np.pi, 400)
x = 2000 + 500*np.cos(s)
y = 2000 + 500*np.sin(s)
init = np.array([x, y]).T
print("Starting")

snake = active_contour(gaussian(img, 10),
                       init) # alpha=0.015, beta=10, gamma=0.001)
print("finish")
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img )
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])



