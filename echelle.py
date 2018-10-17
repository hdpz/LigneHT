import numpy as np

from skimage import io
import pylab as plt


def pixelOrange(pix):
    '''Renvoies true si la couleur du pixel est proche du orange'''
    red, green, blue = pix[0], pix[1], pix[2]
    if red > 130 and blue < 100 and green < 100:
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


def listePixOrange(img):
    ''' Renvoie la liste des coordonnées pix des pixels oranges'''
    X = []
    Y = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if pixelOrange(img[y, x]):
                X.append(x)
                Y.append(y)
    return(X, Y)


def centreEtRayons(X, Y):
    centreX = np.mean(X)
    centreY = np.mean(Y)
    rayons = []
    for i in range(len(X)):

        rayon = np.sqrt((X[i]-centreX)**2+(Y[i]-centreY)**2)
        rayons.append(rayon)
    return rayons, centreX, centreY


def calculeCercle(X, Y):
    rayons, centreX, centreY = centreEtRayons(X, Y)
    # nouvelle iteration pour eliminer les points trop loins
    for i in range(len(rayons)):
        seuil = np.mean(rayons)+2*np.std(rayons)
        if rayons[i] > seuil:
            X.pop(i)
            Y.pop(i)
            print('jai pope')
    rayons, centreX, centreY = centreEtRayons(X, Y)
    return(np.max(rayons), centreX, centreY)


def afficheImageAvecPixOrange(img):
    ''' Affiche les pixels oranges sur l'image d'origine '''
    X, Y = listePixOrange(img)
    R, x_C, y_C = calculeCercle(X, Y)
    t = np.linspace(0, 2*np.pi, 400)
    x = R*np.cos(t)+x_C
    y = R*np.sin(t)+y_C
    plt.imshow(img)
    plt.plot(X, Y, '+b', linewidth=3)
    plt.plot(x, y)
    plt.show()
    return(X, Y)


if __name__ == '__main__':
    img = io.imread('Images/face_bouchon.jpg')
    # print(pixelOrange([222, 100, 50]))
    # comptePixOrange(img)
    X, Y = afficheImageAvecPixOrange(img)
    # print(rayonCercle(X, Y))
