import numpy as np

from skimage import io
import pylab as plt


RAYON_BOUCHON = 2
DISTANCE_CERCLES = 134


def pixelOrange(pix):
    '''Renvoies true si la couleur du pixel est proche du orange'''
    red, green, blue = pix[0], pix[1], pix[2]
    if red > 120 and blue < 100 and green < 80:
        return True
    return False


def pixelRouge(pix):
    '''Renvoies true si la couleur du pixel est proche du orange'''
    red, green, blue = pix[0], pix[1], pix[2]
    if red > 100 and blue < 100 and green < 50:
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


def listePix(img, couleur):
    ''' Renvoie la liste des coordonnées pix des pixels de couleur donnee'''
    X = []
    Y = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if couleur == 'orange':
                if pixelOrange(img[y, x]):
                    X.append(x)
                    Y.append(y)
            if couleur == 'rouge':
                if pixelRouge(img[y, x]):
                    X.append(x)
                    Y.append(y)
    if len(X) == 0:
        raise Exception('Could not find any matching pixel')
    return(X, Y)


def centreEtRayons(X, Y):
    '''calcule le centre du cercle et les rayons de chacun des points a ce cercle'''
    centreX = np.mean(X)
    centreY = np.mean(Y)
    rayons = []
    for i in range(len(X)):
        rayon = np.sqrt((X[i]-centreX)**2+(Y[i]-centreY)**2)
        rayons.append(rayon)
    return rayons, centreX, centreY


def calculeCercle(X, Y):
    '''calcule iterativement le cercle du bouchon et ses coordonnees en eliminant les points extremes'''
    rayons, centreX, centreY = centreEtRayons(X, Y)
    # nouvelle iteration pour eliminer les points trop loins
    indexAEnlever = []
    for i in range(len(rayons)):
        seuil = np.mean(rayons)+2*np.std(rayons)
        if rayons[i] > seuil:
            indexAEnlever.append(i)
    X = [X[i]for i in range(len(X)) if i not in indexAEnlever]
    Y = [Y[i]for i in range(len(Y)) if i not in indexAEnlever]
    rayons, centreX, centreY = centreEtRayons(X, Y)
    return(np.max(rayons), centreX, centreY)


def echelleAPartirDuBouchon(img):
    '''calcule l'echelle de l'image ainsi que le centre du bouchon'''
    X, Y = listePix(img, 'orange')
    R, centreX, centreY = calculeCercle(X, Y)
    echelle = RAYON_BOUCHON / R
    return echelle, centreY


def diviserImgEnDeux(img):
    '''divise l'image en deux pour traiter chaque cercle independemment'''
    partieGauche = img[:, :int(img.shape[1]/2)]
    partieDroite = img[:, int(img.shape[1]/2):]
    return partieGauche, partieDroite


def echelleAPartirDessins(img, display=False):
    '''calcule l'echelle de l'image ainsi que le centre vertical du cercle gauche'''
    partieGauche, partieDroite = diviserImgEnDeux(img)
    X, Y = listePix(partieGauche, 'rouge')
    R, centreXGauche, centreYGauche = calculeCercle(X, Y)
    if display:
        t = np.linspace(0, 2*np.pi, 400)
        x = R*np.cos(t)+centreXGauche
        y = R*np.sin(t)+centreYGauche
        plt.imshow(partieGauche)
        plt.plot(X, Y, '+b', linewidth=3)
        plt.plot(x, y)
        plt.show()
    X, Y = listePix(partieDroite, 'rouge')
    R, centreXDroite, centreYDroite = calculeCercle(X, Y)
    # compense la division de limage en deux
    centreXDroite = centreXDroite + img.shape[1]/2
    distanceEntreCercles = np.sqrt(
        (centreXDroite-centreXGauche)**2+(centreYDroite-centreYGauche)**2)
    echelle = DISTANCE_CERCLES/distanceEntreCercles
    return echelle, centreYGauche


def afficheImageAvecPixOrange(img):
    ''' Affiche les pixels oranges sur l'image d'origine '''
    X, Y = listePix(img, 'orange')
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
    img = io.imread('Images/face_cercles.jpg')
    echelle, centreYGauche = echelleAPartirDessins(img)
