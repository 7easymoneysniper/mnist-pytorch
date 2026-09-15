import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. Semilla para reproducibilidad
def fijar_semilla(seed=42):
    torch.manual_seed(seed)


# 2. Detectar dispositivo disponible (GPU > CPU)
def obtener_dispositivo():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# 3. Preparar los datos: convierte las imágenes a tensores y las normaliza
def obtener_datos(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 4. Definir los modelos

# Red neuronal simple (MLP): más rápida pero menos precisa
class RedSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.capas = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.capas(x)


# Red neuronal convolucional (CNN): más lenta pero más precisa
class RedConvolucional(nn.Module):
    def __init__(self):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.capas(x)


def obtener_modelo(nombre, dispositivo):
    if nombre == "cnn":
        modelo = RedConvolucional()
    else:
        modelo = RedSimple()
    return modelo.to(dispositivo)


# 5. Evaluar precisión sobre un conjunto de datos
def evaluar(modelo, loader, dispositivo):
    correctos = 0
    total = 0
    perdida_total = 0.0
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
            salida = modelo(imagenes)
            perdida = nn.CrossEntropyLoss()(salida, etiquetas)
            perdida_total += perdida.item() * etiquetas.size(0)
            _, prediccion = torch.max(salida, 1)
            total += etiquetas.size(0)
            correctos += (prediccion == etiquetas).sum().item()
    return 100.0 * correctos / total, perdida_total / total


# 6. Entrenamiento
def entrenar(modelo, train_loader, test_loader, dispositivo, epocas, lr):
    funcion_perdida = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr)

    mejor_precision = 0.0
    for epoca in range(1, epocas + 1):
        modelo.train()
        perdida_acumulada = 0.0
        total = 0
        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
            optimizador.zero_grad()
            salida = modelo(imagenes)
            perdida = funcion_perdida(salida, etiquetas)
            perdida.backward()
            optimizador.step()
            perdida_acumulada += perdida.item() * etiquetas.size(0)
            total += etiquetas.size(0)

        perdida_entrenamiento = perdida_acumulada / total
        modelo.eval()
        precision_test, perdida_test = evaluar(modelo, test_loader, dispositivo)
        print(f"Época {epoca:2d}/{epocas} | Pérdida train: {perdida_entrenamiento:.4f} | "
              f"Pérdida test: {perdida_test:.4f} | Precisión test: {precision_test:.2f}%")

        if precision_test > mejor_precision:
            mejor_precision = precision_test
            torch.save(modelo.state_dict(), "mejor_modelo.pt")

    print(f"\nMejor precisión en test: {mejor_precision:.2f}%")
    print(f"Modelo guardado en: mejor_modelo.pt")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de MNIST con PyTorch")
    parser.add_argument("--modelo", choices=["mlp", "cnn"], default="cnn",
                        help="Tipo de red: mlp (rápida) o cnn (precisa)")
    parser.add_argument("--epocas", type=int, default=3, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=0.001, help="Tasa de aprendizaje")
    parser.add_argument("--batch", type=int, default=64, help="Tamaño de lote")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    fijar_semilla(args.seed)
    dispositivo = obtener_dispositivo()
    print(f"Dispositivo: {dispositivo}")

    train_loader, test_loader = obtener_datos(args.batch)
    modelo = obtener_modelo(args.modelo, dispositivo)
    print(f"Modelo: {type(modelo).__name__} | Parámetros: {sum(p.numel() for p in modelo.parameters()):,}")

    inicio = time.time()
    entrenar(modelo, train_loader, test_loader, dispositivo, args.epocas, args.lr)
    print(f"Tiempo total: {time.time() - inicio:.1f} s")


if __name__ == "__main__":
    main()