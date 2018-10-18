import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from echelle import echelleAPartirDuBouchon, echelleAPartirDessins
from sobel import calculeHauteur

HAUTEUR_BOUCHON = 29
HAUTEUR_CERCLE_GAUCHE = 75


if __name__ == '__main__':
    imgGris = io.imread('Images/face_cercles_160.jpg', as_gray=True)
    imgCouleur = io.imread('Images/face_cercles_160.jpg')

    echelle, centreY = echelleAPartirDessins(imgCouleur)
    print('echelle de', echelle)

    hauteurFil = calculeHauteur(imgGris, 0.08, display=True)
    print(hauteurFil)
    hauteurReelle = (centreY-hauteurFil)*echelle + HAUTEUR_CERCLE_GAUCHE
    print('hauteur reelle de', hauteurReelle)
