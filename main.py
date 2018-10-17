import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from echelle import echelleAPartirDuBouchon
from sobel import calculeHauteur

HAUTEUR_BOUCHON = 29

if __name__ == '__main__':
    imgGris = io.imread('Images/face_bouchon.jpg', as_gray=True)
    imgCouleur = io.imread('Images/face_bouchon.jpg')

    echelle, centreY = echelleAPartirDuBouchon(imgCouleur)
    print('echelle de', echelle)

    hauteurFil = calculeHauteur(imgGris, 0.1)

    hauteurReelle = (centreY-hauteurFil)*echelle + HAUTEUR_BOUCHON
    print('hauteur reelle de', hauteurReelle)
