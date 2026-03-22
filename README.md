# Carburant Predict

Projet de prediction du prix des carburants en France a partir des donnees publiques du gouvernement.

## Fichiers du projet

- `carburant_prediction.ipynb` : notebook principal de preparation, entrainement et prediction.
- `interface.py` : interface Gradio pour tester le modele sur un departement, un carburant et un horizon.
- `requirements.txt` : dependances Python du projet.

## Environnement virtuel

Le projet utilise un environnement virtuel local nomme `carburant`.

### Creation manuelle

```powershell
python -m venv carburant
.\carburant\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name carburant --display-name "Python (carburant)"
```

## Utilisation

### Ouvrir le notebook

```powershell
.\carburant\Scripts\Activate.ps1
python -m jupyter lab carburant_prediction.ipynb
```

Dans Jupyter ou VS Code, selectionner ensuite le kernel `Python (carburant)`.

### Lancer l'interface

```powershell
.\carburant\Scripts\Activate.ps1
python interface.py
```

L'interface permet de tester directement le modele sauvegarde avec les donnees 2026 de l'archive officielle.

## Remarques

- Le notebook est configure pour utiliser le kernel `carburant`.
- Au premier lancement, l'interface recharge l'archive 2026 puis met en cache les donnees pour accelerer les essais suivants.
- Les emojis ont ete retires du notebook et de l'interface pour garder un rendu sobre.
