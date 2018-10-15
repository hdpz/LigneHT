import numpy as np
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt, laplace
from skimage import io

img = io.imread('Images/face_fil.jpg', as_gray=True)

edge_roberts = roberts(img)
edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)
edge_laplace = laplace(img)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                       figsize=(20, 20))


ax[0].imshow(edge_sobel, cmap=plt.cm.gray)
ax[0].set_title('Sobel Edge Detection')


for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
