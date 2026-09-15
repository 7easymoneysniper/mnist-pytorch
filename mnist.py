import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Preparar los datos: convierte las imágenes a tensores y las normaliza
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 2. Definir el modelo: una red neuronal simple
class RedSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.capas = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.capas(x)

modelo = RedSimple()

# 3. Función de pérdida y optimizador
funcion_perdida = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

# 4. Entrenamiento
epocas = 3
for epoca in range(epocas):
    for imagenes, etiquetas in train_loader:
        optimizador.zero_grad()
        salida = modelo(imagenes)
        perdida = funcion_perdida(salida, etiquetas)
        perdida.backward()
        optimizador.step()
    print(f"Época {epoca+1}/{epocas} - Pérdida: {perdida.item():.4f}")

# 5. Evaluación
correctos = 0
total = 0
with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        salida = modelo(imagenes)
        _, prediccion = torch.max(salida, 1)
        total += etiquetas.size(0)
        correctos += (prediccion == etiquetas).sum().item()

print(f"Precisión en test: {100 * correctos / total:.2f}%")
