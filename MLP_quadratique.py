import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

# ===================== Dataset =====================
class ImageRegressionDataset(Dataset):
    def __init__(self, excel_path, image_folder):
        self.df = pd.read_excel(excel_path)
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        params = self.df.iloc[idx][['a', 'b', 'c']].values.astype(np.float32)
        img_path = os.path.join(self.image_folder, f"img_{idx:03}.png")
        image = Image.open(img_path).convert('L')  
        image = self.transform(image)
        return torch.tensor(params), image

# ===================== MLP Model =====================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128*128),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ===================== Image generation =====================
def generate_image(a, b, c, idx, folder="images"):
    os.makedirs(folder, exist_ok=True)
    x = np.linspace(-10, 10, 100)
    y = a * x**2 + b * x + c
    plt.figure(figsize=(2, 2), dpi=32)
    plt.plot(x, y, color='black', linewidth=3)
    plt.ylim(-200, 200)
    plt.axis('off')
    plt.tight_layout()
    path = os.path.join(folder, f"img_{idx:03}.png")
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return path

def generate_excel(param_list, path='params.xlsx'):
    df = pd.DataFrame(param_list, columns=['a', 'b', 'c'])
    df.to_excel(path, index=False)

def generate_all_images_from_excel(excel_path='params.xlsx', folder='images'):
    df = pd.read_excel(excel_path)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        generate_image(row['a'], row['b'], row['c'], idx, folder)

# ===================== Training =====================
def train_model(dataset_path='params.xlsx', image_folder='images',
                save_path='mlp_weights.pth', epochs=50, batch_size=8):
    dataset = ImageRegressionDataset(dataset_path, image_folder)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            targets = targets.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                targets = targets.view(inputs.size(0), -1)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ===================== Inference =====================
def main_inference(a, b, c, model_path='mlp_weights.pth', output_folder='predictions'):
    os.makedirs(output_folder, exist_ok=True)
    model = MLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([[a, b, c]], dtype=torch.float32)
        output = model(input_tensor).view(128, 128).numpy()
        output[output < 0.8] *= 0.2
        plt.imshow(output, cmap='gray')
        plt.axis('off')
        image_path = os.path.join(output_folder, f"prediction_a{a}_b{b}_c{c}.png")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    print(f"Image saved to {image_path}")
