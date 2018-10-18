import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from echelle import echelleAPartirDuBouchon, echelleAPartirDessins
from sobel import calculeHauteur

HAUTEUR_BOUCHON = 29
HAUTEUR_CERCLE_GAUCHE = 78


if __name__ == '__main__':
    imgGris = io.imread('Images/face_cercles.jpg', as_gray=True)
    imgCouleur = io.imread('Images/face_cercles.jpg')

    echelle, centreY = echelleAPartirDessins(imgCouleur)
    print('echelle de', echelle)
    print(centreY)

    hauteurFil = calculeHauteur(imgGris, 0.1)
    print(hauteurFil)
    hauteurReelle = (centreY-hauteurFil)*echelle + HAUTEUR_CERCLE_GAUCHE
    print('hauteur reelle de', hauteurReelle)
