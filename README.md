# 🚀 Portfolio — Atillio HOUNGUE
### Data Scientist & Data Analyst

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-a126OPS-orange?logo=huggingface)](https://huggingface.co/a126OPS)
[![GitHub](https://img.shields.io/badge/GitHub-a126OPS-black?logo=github)](https://github.com/a126OPS)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Atillio%20Gautier-blue?logo=linkedin)](https://www.linkedin.com/in/atillio-houngue-245715331)
[![Deploy](https://img.shields.io/badge/Deploy-GitHub%20Pages-green?logo=github)](https://a126ops.github.io/projet_public/)

---

## 📋 Présentation

Portfolio personnel de **Atillio HOUNGUE**, Alternant Data Ingénieur chez Quai des Notaires.  
Ce site présente mes projets Machine Learning déployés en production sur Hugging Face, mon parcours professionnel et mes compétences techniques.

---

## 🧠 Projets ML en production

| Projet | Modèle HF | Space Gradio | Description |
|--------|-----------|--------------|-------------|
| 🚗 Prix Voiture | [`Car_Predict`](https://huggingface.co/a126OPS/Car_Predict) | [`car-price-predictor-demo`](https://huggingface.co/spaces/a126OPS/car-price-predictor-demo) | Prédiction du prix d'un véhicule d'occasion |
| 🏠 Immo Saône-et-Loire | [`prediction_immo_soane_et_loire`](https://huggingface.co/a126OPS/prediction_immo_soane_et_loire) | [`prediction_immo_soane_et_loirePS`](https://huggingface.co/spaces/a126OPS/prediction_immo_soane_et_loirePS) | Estimation prix immobilier (71) |
| ⛽ Prix Carburant | [`carburant_price_predict`](https://huggingface.co/a126OPS/carburant_price_predict) | [`carburant_predict`](https://huggingface.co/spaces/a126OPS/carburant_predict) | Prédiction J+7 du prix à la pompe |
| ⚡ Conso Électrique | [`conso-energie-predict`](https://huggingface.co/a126OPS/conso-energie-predict) | [`conso_energie_predict`](https://huggingface.co/spaces/a126OPS/conso_energie_predict) | Estimation consommation résidentielle (MAE 307 kWh/an) |

---

## 🗂️ Structure du projet

```text
projet_public/
│
├── atillio_portfolio.html   # Source locale du portfolio
├── docs/
│   ├── index.html           # Version publiée sur GitHub Pages
│   └── merci.html           # Page de confirmation après soumission du formulaire
└── README.md                # Ce fichier
```

---

## 🛠️ Stack technique du site

- **HTML5 / CSS3 / JavaScript vanilla** — zéro dépendance frontend
- **Fonts** : Syne (titres) + IBM Plex Mono (code/labels) + IBM Plex Sans (corps)
- **Formulaire** : [FormSubmit](https://formsubmit.co/) — sans backend
- **Déploiement** : GitHub Pages
- **Intégrations** : Hugging Face Spaces (iframes Gradio)

---

## 🚀 Déploiement local

```bash
# Cloner le repo
git clone https://github.com/a126OPS/projet_public.git
cd projet_public

# Lancer un serveur local (Python)
python -m http.server 8000

# Ouvrir dans le navigateur
open http://localhost:8000
```

---

## 📦 Mise à jour des modèles HF

Pour mettre à jour un modèle depuis Python :

```python
from huggingface_hub import HfApi

api = HfApi(token="hf_xxxxxxxxxxxx")

api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id="a126OPS/conso-energie-predict",
    repo_type="model"
)
```

---

## 🔄 Architecture de déploiement des modèles

```
Entraînement local (Jupyter / VSCode)
          ↓
Upload model.joblib → HF Model Repo
          ↑ hf_hub_download()
Space Gradio (app.py)
          ↑ iframe embed
Portfolio (index.html)
          ↑ fetch/API
Intégration portfolio (futur)
```

---

## 📬 Contact

- **Email** : atilliohoungue@gmail.com
- **Tél** : 07 45 09 28 92
- **Localisation** : Autun (71400) — Mobile toute la France
- **Disponibilité** : Alternance en cours (oct. 2025) · Télétravail ou présentiel

---

## 📄 Licence

Ce portfolio est open source — tu peux t'en inspirer librement.  
Les modèles ML sont sous licence **MIT**.

---

*Développé par Atillio HOUNGUE — 2026*
