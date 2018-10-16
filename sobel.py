import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt, laplace, sobel_v
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
    X, Y = listing(mat)
    a = [ Y.count(i) for i in range(len(Y))]
    return(a)


def pics(a):
    ind = []
    seuil = np.mean(a)+np.std(a)
    for i in range(len(a)):
        if a[i] < seuil :
            a[i] = 0
        else : 
            ind.append(i)
    return(a, ind)

def mesure_hauteur(mat):
    a = density(mat)
    a, ind = pics(a)
    plt.plot(a)
    plt.axis([0, 4160, 0, 50])
    plt.grid()
    plt.show()
    b = ind
    MIN = min(b)
    MAX = max(b)
    print(MAX-MIN)
    return(MAX-MIN)
    
def hauteur_fil(haut_ref_pix, haut_ref_real, haut_fil_pix):
    return((haut_ref_real*haut_fil_pix)/haut_ref_pix)

if __name__ == '__main__':

    img = io.imread('Images/face_fil.jpg', as_gray=True)
    edge_sobel = sobel(img)

    contraste = augmentContrast(edge_sobel, 0.1)   
    
    
    rogne1 = rogner(contraste, 1000, 3600,  200, 2400)
    rogne2 = rogner(contraste, 500, contraste.shape[1], 0, contraste.shape[0])
    rogne3 = rogner(contraste, 500, int((contraste.shape[1]-500)/2), 0, contraste.shape[0]) 
    rogne = rogne3

    X, Y = listing(rogne)
    z = fitting_parabole(X, Y)

    t = np.linspace(0, rogne.shape[1])
    par = np.polyval(z, t)
    MIN = min_parabole(z)

    
    h =  mesure_hauteur(rogne3)



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
    
    print('Hauteur sous fil : '+str(hauteur_fil(h, 199.5, MIN[1])))

