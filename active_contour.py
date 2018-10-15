import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import io


img = io.imread('Images/face_fil.jpg', as_gray=True)

x = np.linspace(813, 3697, 400)
y = np.linspace(465, 377, 400)
init = np.array([x, y]).T

snake = active_contour(gaussian(img, 3), init,
                       bc='fixed', w_line=-10, w_edge=10, max_iterations=100)

fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])


plt.show()
