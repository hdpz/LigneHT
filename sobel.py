import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt, laplace
from skimage import io

img = io.imread('Images/face_fil.jpg', as_gray=True)

edge_sobel = sobel(img)


def augmentContrast(mat, seuil):
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if mat[y][x] > seuil:
                mat[y][x] = 1
            else:
                mat[y][x] = 0
    return mat


def rogner(mat, xMin, xMax, yMin, yMax):
    matRogne = mat[yMin:yMax, xMin:xMax]
    print(matRogne.shape)
    return matRogne

def listing(mat):
    X = []
    Y = []
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if mat[y][x] == 1 :
                X.append(x)
                Y.append(y)
    return(X, Y)

def fitting_parabole(X, Y):
    z = np.polyfit(X, Y, 2)
    return(z)

def min_parabole(z):
    return(-z[1]/(2*z[0]), np.polyval(z, [-z[1]/(2*z[0])])[0])
    
    
contraste = augmentContrast(edge_sobel, 0.1)
print(contraste.shape)



rogne = rogner(contraste, 1000, 3600,  200, 2400)
X, Y = listing(rogne)
z = fitting_parabole(X, Y)

t = np.linspace(0, rogne.shape[1])
par = np.polyval(z, t)
MIN = min_parabole(z)


fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                       figsize=(20, 20))


ax[0].imshow(rogne, cmap=plt.cm.gray)
ax[0].plot(t, par)
ax[0].plot(MIN[0], MIN[1], '+r', linewidth = 3)
ax[0].set_title('Sobel Edge Detection')


for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()


