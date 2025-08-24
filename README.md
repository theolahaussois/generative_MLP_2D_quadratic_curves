#  Generative MLP 2D Quadratic Curves

Ce repository contient un projet de génération et reconstruction d’images de courbes quadratiques 2D à l’aide d’un **MLP (Multi-Layer Perceptron)**.  
Le modèle pré-entraîné est fourni pour exécuter des inférences rapides sans réentraînement.

---

##  Contenu du dépôt

| Fichier / Dossier | Description |
|------------------|-------------|
| `MLP_quadratique.py` | Module Python principal avec la définition du dataset, du MLP, fonctions d’entraînement et d’inférence |
| `main.py` | Script CLI pour lancer entraînement ou prédiction |
| `images/` | Images générées à partir des paramètres de courbes |
| `predictions/` | Dossier où les images générées par le modèle sont sauvegardées |
| `params.xlsx` | Fichier Excel contenant les paramètres `(a, b, c)` pour générer les courbes |
| `mlp_weights.pth` | Poids du modèle entraîné (prêt à l’usage) |
| `mlp_checkpoints.pth` | Checkpoints d’entraînement optionnels |
| `__pycache__/` | Cache Python (à ignorer dans le git push) |

---

##  Installation


```bash
git clone https://github.com/theolahaussois/generative_MLP_2D_quadratic_curves.git
cd generative_MLP_2D_quadratic_curves
pip install numpy pandas matplotlib pillow tqdm scikit-learn torch torchvision
python main.py --predict --coeffs 4 8 3 --out predictions
