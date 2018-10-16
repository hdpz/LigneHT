#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:41:36 2018

@author: jb
"""
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
"""
import numpy as np

from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny


import matplotlib.pyplot as plt
from matplotlib import cm

from skimage import io

# Line finding using the Probabilistic Hough Transform
image = io.imread("Images/face_fil.jpg", as_gray = True)
edges = canny(image, 0.5)
lines = probabilistic_hough_line(edges, threshold=2, line_length=20, line_gap=2)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()

