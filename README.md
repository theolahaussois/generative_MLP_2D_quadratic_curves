# Generative MLP 2D Quadratic Curves

Ce projet permet de **générer et reconstruire des images de courbes quadratiques 2D** à l’aide d’un **MLP (Multi-Layer Perceptron)** en PyTorch.  
Le modèle pré-entraîné est inclus pour permettre des inférences rapides sans réentraînement.

---

## Contenu du dépôt

| Fichier / Dossier        | Description |
|--------------------------|-------------|
| `MLP_quadratique.py`     | Définition du dataset, du MLP et des fonctions d’entraînement et d’inférence |
| `main.py`                | Script CLI pour lancer l’entraînement ou la prédiction |
| `predictions/`           | Dossier où sont sauvegardées les images générées par le modèle |
| `mlp_weights.pth`        | Poids du modèle pré-entraîné (prêt à l’usage) |
| `mlp_checkpoints.pth`    | Checkpoints d’entraînement intermédiaires |
| `__pycache__/`           | Cache Python |

---

## Installation

Cloner le dépôt et installer les dépendances :

```bash
git clone https://github.com/theolahaussois/generative_MLP_2D_quadratic_curves.git
cd generative_MLP_2D_quadratic_curves
pip install -r requirements.txt
```

---

## Utilisation

```bash
python main.py --coeffs 4 8 3 --out predictions/
```


---

eckpoint mlp_weights.pth
```
