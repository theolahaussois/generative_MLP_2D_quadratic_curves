# generative_MLP_2D_quadratic_curves

# Generative MLP for 2D Quadratic Curves

This project implements a **Multi-Layer Perceptron (MLP)** to generate images of quadratic curves from their parameters `(a, b, c)`.

## 📂 Contents
- `MLP_quadratique.ipynb` : main notebook (data generation, training, inference)
- `images/` : dataset of generated curves
- `params.xlsx` : parameters of curves
- `predictions/` : model predictions
- `mlp_weights.pth` : trained model

## 🚀 Usage
1. Generate dataset (parameters + images)  
2. Train the MLP on quadratic curves  
3. Predict new curves from given `(a, b, c)`  

Example:
```python
from MLP_quadratique import main_inference
main_inference(4, 8, 3, output_folder="predictions")
