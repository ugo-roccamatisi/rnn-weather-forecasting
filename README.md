# TP : Réseaux récurrents pour la prévision météo

Travaux pratiques de deep learning (PyTorch) : prise en main des RNN, GRU et LSTM, puis prévision de séries temporelles météorologiques mesurées à l'aéroport d'Orly.

## Contenu

- Rappels sur les RNN simples, déroulement dans le temps, équations et intuition des GRU et LSTM
- Préparation des données SYNOP (pression, variation de pression, direction et vitesse du vent, température, point de rosée, humidité), une mesure toutes les 3 heures
- Premier modèle de prévision à un pas et comparaison à la persistance
- Prévision à long terme (plusieurs pas), modèles plus larges
- Entraînement amélioré : scheduled sampling avec rétropropagation à travers la génération

## Principaux résultats

- Un GRU prédit bien la mesure suivante (3 h) : erreur réduite d'environ 35 % par rapport à la persistance.
- En prévision à plusieurs pas avec teacher forcing, un petit modèle fait moins bien que la répétition du dernier jour observé dès 24 h ; il faut plus de capacité pour la dépasser.
- Le scheduled sampling réduit l'erreur en prévision libre de 15 % à architecture égale : environ 2,5 °C d'erreur sur la température à 24 h et 3,1 °C à 45 h.

## Structure

```
ODL_lab_recurrent_meteo_2026.ipynb   notebook du TP (exécuté)
meteo-train.py.npy                   données d'entraînement
meteo-test.py.npy                    données de test
requirements.txt
```

## Lancer le notebook

```bash
pip install -r requirements.txt
jupyter notebook ODL_lab_recurrent_meteo_2026.ipynb
```

## Données

Données SYNOP essentielles de l'OMM (station d'Orly), disponibles en open data sur [OpenDataSoft](https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm/information/), filtrées et préparées pour le TP.
