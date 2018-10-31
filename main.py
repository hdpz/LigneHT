import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from echelle import echelleAPartirDuBouchon, echelleAPartirDessins
from sobel import calculeHauteur
from scoring import computeModels

Confs = {'face_cercles': {'hCercle': 75, 'dCercles': 134}, 'face_cercles_130': {'hCercle': 75, 'dCercles': 134},
         'face_cercles_160': {'hCercle': 75, 'dCercles': 134}, 'dehors': {'hCercle': 39, 'dCercles': 294}}


def chargerImages(path='Images/', formate='.jpg', listeImages=['face_cercles',
                                                               'face_cercles_130', 'face_cercles_160', 'dehors']):
    '''Renvois deux dicts d'images en couleur et grises'''
    imagesCouleur = {nom: io.imread(path+nom+formate)
                     for nom in listeImages}
    imagesGris = {nom: io.imread(path+nom+formate, as_gray=True)
                  for nom in listeImages}
    return imagesCouleur, imagesGris


if __name__ == '__main__':
    imagesCouleur, imagesGris = chargerImages()

    for nom in imagesCouleur:
        # Calcul de l'echelle de l'image
        echelle, centreY = echelleAPartirDessins(
            imagesCouleur[nom], Confs[nom]['dCercles'])
        print(nom + ' a une echelle de', echelle)

        # Application des differents modeles a l'image en nuance de gris
        matricesGradient = computeModels(imagesGris[nom])
        for model in matricesGradient:
            print('Doing model ' + model)
            hauteurFil = calculeHauteur(
                matricesGradient[model], 0.08, display=False)

            hauteurReelle = (centreY-hauteurFil) * \
                echelle + Confs[nom]['hCercle']
            print(nom + ' et ' + model + ' a une hauteur reelle de', hauteurReelle)
