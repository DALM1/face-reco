# Face-Track

## Description
**Face-Track** est un projet Python avancé de détection de visages en temps réel utilisant OpenCV et PyQt5. Le programme capture les visages depuis une caméra en direct, affiche un carrousel des visages détectés, et offre des fonctionnalités interactives pour visualiser, rechercher et télécharger les visages. Le projet intègre également des capacités de sécurité avancées comme le hachage des visages et le chiffrement des images.

## Fonctionnalités
- **Détection de visages en temps réel** depuis une webcam.
- **Carrousel dynamique** affichant tous les visages détectés.
- **Affichage en grand format** des visages cliqués dans le carrousel.
- **Téléchargement des visages** sous forme d'images chiffrées.
- **Hachage SHA256** des visages avec affichage du hash sous chaque image.
- **Recherche de visages par hash** : possibilité d'entrer un hash dans une barre de recherche pour retrouver un visage spécifique.
- **Chargement d'une photo externe** pour rechercher un visage déjà capturé.
- **Copie rapide du hash** d'une image cliquée pour une utilisation ultérieure.

## Prérequis

- Python 3.x
- OpenCV
- PyQt5
- Cryptography
- Scikit-learn (pour l'amélioration de la reconnaissance des visages)

## Installation

1. Clonez ce repository ou téléchargez les fichiers source
   ```bash
   git clone https://github.com/DALM1/face-reco.git
