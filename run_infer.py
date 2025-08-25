
from MLP_quadratique import main_inference

# Coefficients pour la prédiction
a, b, c = 4, 8, 3

main_inference(
    a, b, c,
    model_path='mlp_weights.pth',
    output_folder='predictions'
)
