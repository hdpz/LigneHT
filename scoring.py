import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt, laplace
from skimage import io, feature


def computeModels(img):
    '''Calcule les resultats des differentes methodes de detection d'objets'''
    results = {'Image original': IMG}
    results['Canny'] = feature.canny(img, sigma=1)
    results['Roberts'] = roberts(img)
    results['Sobel'] = sobel(img)
    results['Scharr'] = scharr(img)
    results['Prewitt'] = prewitt(img)
    results['Laplace'] = laplace(img)
    return results


def displayImages(results):
    '''arrange les resultats dans une matrice pour que ce soit tout beau'''
    fig, plots = plt.subplots(nrows=3, ncols=3, figsize=(10, 15),
                              sharex=True, sharey=True)

    ordre = ['Image original', 'Canny', 'Roberts',
             'Sobel', 'Scharr', 'Prewitt', 'Laplace']
    i, j = 0, 1
    for titre in ordre:
        print('adding en ', i, j, titre)

        plots[i][j].imshow(results[titre], cmap=plt.cm.gray)
        plots[i][j].set_title(titre, fontsize=20)
        plots[i][j].axis('off')
        if i == 0 or j == 2:
            i += 1
            j = 0
        else:
            j += 1

    plots[0, 0].axis('off')
    plots[0, 2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    IMG = io.imread('Images/face_fil.jpg', as_gray=True)
    results = computeModels(IMG)
    displayImages(results)
