import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt, laplace, sobel_v
from skimage import io


def augmentContrast(edgeSobel, seuil, display=False):
    '''augmente le contraste avec un seuil de luminosite'''
    contraste = np.copy(edgeSobel)
    for y in range(edgeSobel.shape[0]):
        for x in range(edgeSobel.shape[1]):
            if edgeSobel[y, x] > seuil:
                contraste[y, x] = 1
            else:
                contraste[y, x] = 0
    if display:
        fig, plot = plt.subplots()
        plot.imshow(contraste, cmap=plt.cm.gray)
        plot.axis('off')
        plt.show()
    return contraste


def rogner(mat, xMin, xMax, yMin, yMax):
    '''rogne la photo pour n'avoir que la corde pour fitter le polynome'''
    matRogne = mat[yMin:yMax, xMin:xMax]
    return matRogne


def listing(mat):
    '''extrait les listes de coordonnees des points d'interfaces'''
    X = []
    Y = []
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if mat[y][x] == 1:
                X.append(x)
                Y.append(y)
    return(X, Y)


def fitting_parabole(X, Y):
    '''fit un polynome sur les listes de coordonnees'''
    z = np.polyfit(X, Y, 2)
    return(z)


def min_parabole(z):
    '''calcule le min du polynome'''
    return(-z[1]/(2*z[0]), np.polyval(z, [-z[1]/(2*z[0])])[0])


def trouveryMax(mat):
    '''methode qui ne marche pas a cause des points parasites'''
    yMax = 0
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if mat[y, x] == 1 and y > yMax:
                yMax = y
    return yMax


def density(mat):
    '''calcule le nombre de points lumineux par ligne'''
    X, Y = listing(mat)
    counter = [Y.count(i) for i in range(len(Y))]
    return(counter)


def pics(a):
    '''filtre les lignes avec un trop faible nombre de points'''
    ind = []
    seuil = np.mean(a)+np.std(a)
    for i in range(len(a)):
        if a[i] < seuil:
            a[i] = 0
        else:
            ind.append(i)
    return(a, ind)


def mesure_hauteur(mat, display=False):
    a = density(mat)
    a, ind = pics(a)
    if display:
        plt.plot(a)
        plt.axis([0, 4160, 0, 50])
        plt.grid()
        plt.show()
    b = ind
    MIN = min(b)
    MAX = max(b)
    return(MAX-MIN)


def hauteur_fil(haut_ref_pix, haut_ref_real, haut_fil_pix):
    return((haut_ref_real*haut_fil_pix)/haut_ref_pix)


def calculeHauteur(mat, seuilContraste, display=False):
    contraste = augmentContrast(edge_sobel, seuilContraste)
    rogne1 = rogner(contraste, 1000, 3600,  200, 2400)
    rogne2 = rogner(contraste, 500, contraste.shape[1], 0, contraste.shape[0])
    rogne3 = rogner(contraste, 500, int(
        (contraste.shape[1]-500)/2), 0, contraste.shape[0])
    rogne = rogne3
    X, Y = listing(rogne)
    z = fitting_parabole(X, Y)

    t = np.linspace(0, rogne.shape[1])
    par = np.polyval(z, t)
    MIN = min_parabole(z)

    h = mesure_hauteur(rogne3)
    if display:
        fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                               figsize=(20, 20))

        ax[0].imshow(rogne, cmap=plt.cm.gray)
        ax[0].plot(t, par)
        ax[0].plot(MIN[0], MIN[1], '+r', linewidth=3)
        ax[0].set_title('Sobel Edge Detection')

        for a in ax:
            a.axis('off')

        plt.tight_layout()
        plt.show()

    return hauteur_fil(h, 199.5, MIN[1])


def trouverSeuilContrasteOptimal(edgeSobel, hauteurDuFil):
    meilleurSeuil, plusPetitEcart = 0, 100
    valeurDeSeuils = np.linspace(0.05, 0.1, 10)
    for seuilContraste in valeurDeSeuils:
        hauteurCalculee = calculeHauteur(edgeSobel, seuilContraste)
        ecart = hauteurDuFil-hauteurCalculee
        if ecart < plusPetitEcart:
            plusPetitEcart = ecart
            meilleurSeuil = seuilContraste
        print('Seuil', seuilContraste, 'ecart', ecart)
    return(meilleurSeuil, plusPetitEcart)


if __name__ == '__main__':

    img = io.imread('Images/face_fil_3.jpg', as_gray=True)
    edge_sobel = sobel(img)

    # augmentContrast(edge_sobel, 0.1, True)

    trouverSeuilContrasteOptimal(edge_sobel, 112)
