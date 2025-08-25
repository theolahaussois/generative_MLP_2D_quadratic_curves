
from MLP_quadratique import train_model
import numpy as np
from MLP_quadratique import generate_excel, generate_all_images_from_excel

# -------------------- Configuration --------------------
N = 10  # nombre d'images à générer
excel_path = 'params.xlsx'
image_folder = 'images'

# -------------------- Génération des paramètres --------------------
param_list = np.random.uniform(-5, 5, size=(N, 3))  # N triplets (a,b,c)
generate_excel(param_list, path=excel_path)
print(f"{N} paramètres générés et sauvegardés dans {excel_path}")

# -------------------- Génération des images --------------------
generate_all_images_from_excel(excel_path=excel_path, folder=image_folder)
print(f"Images générées dans le dossier {image_folder}")

train_model(
    dataset_path='params.xlsx',
    image_folder='images',
    save_path='mlp_weights.pth',
    epochs=2,
    batch_size=8
)
