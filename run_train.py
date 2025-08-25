
from MLP_quadratique import train_model

train_model(
    dataset_path='params.xlsx',
    image_folder='images',
    save_path='mlp_weights.pth',
    epochs=2,
    batch_size=8
)
