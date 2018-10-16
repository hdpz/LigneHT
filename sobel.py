import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt, laplace
from skimage import io


def augmentContrast(mat, seuil):
    '''augmente le contraste avec un seuil de luminosite'''
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if mat[y][x] > seuil:
                mat[y][x] = 1
            else:
                mat[y][x] = 0
    return mat


def rogner(mat, xMin, xMax, yMin, yMax):
    '''rogne la photo pour n'avoir que la corde pour fitter le polynome'''
    matRogne = mat[yMin:yMax, xMin:xMax]
    return matRogne


def trouveryMax(mat):
    '''methode qui ne marche pas a cause des points parasites'''
    yMax = 0
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if mat[y, x] == 1 and y > yMax:
                yMax = y
    return yMax


if __name__ == '__main__':

    img = io.imread('Images/face_fil.jpg', as_gray=True)
    edge_sobel = sobel(img)

    contraste = augmentContrast(edge_sobel, 0.1)
    rogne = rogner(contraste, 1000, 3600,  200, 2400)

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                           figsize=(20, 20))

    ax[0].imshow(rogne, cmap=plt.cm.gray)
    ax[0].set_title('Sobel Edge Detection')
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()
