import numpy as np

from skimage import io


def pixelOrange(pix):
    '''Renvoies true si la couleur du pixel est proche du orange'''
    red, green, blue = pix[0], pix[1], pix[2]
    if red > 130:
        return True
    return False


def comptePixOrange(img):
    '''Comptes les pixels oranges dans l'image'''
    nb = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if pixelOrange(img[y, x]):
                nb += 1
    print(nb)


if __name__ == '__main__':
    img = io.imread('Images/face_bouchon.jpg')
    #print(pixelOrange([222, 100, 50]))
    comptePixOrange(img)
