import torch
import torch.nn as nn
import torch.optim as optim
from data.MNIST_dataset import train_loader, test_loader  # Importa los dataloaders de tu archivo MNIST_dataset.py
from models.CVAE import CVAE  # Importa el modelo CVAE de tu archivo CVAE.py

# Define el dispositivo a utilizar (GPU si está disponible, de lo contrario CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparámetros
input_dim = 784  # Dimensión de entrada de las imágenes MNIST (28x28 = 784)
latent_dim = 20  # Dimensión del espacio latente
output_dim = input_dim  # Dimensión de salida (igual a la dimensión de entrada)
hidden_dim = 128  # Dimensión de las capas ocultas (Neuronas en cada capa oculta)
learning_rate = 0.001
num_epochs = 10

# Inicializa el modelo
model = CVAE(input_dim, latent_dim, hidden_dim, output_dim).to(device)

# Define la función de pérdida (Mean Squared Error)
criterion = nn.MSELoss()

# Define el optimizador (Adam)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Función de entrenamiento
def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)  # Aplana las imágenes
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = criterion(recon_batch, data)  # Calcula la pérdida
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    return running_loss / len(train_loader.dataset)

# Función de evaluación
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, 784).to(device)  # Aplana las imágenes
            recon_batch = model(data)
            test_loss += criterion(recon_batch, data).item()  # Suma la pérdida de reconstrucción
    test_loss /= len(test_loader.dataset)
    return test_loss

# Entrenamiento del modelo
for epoch in range(1, num_epochs + 1):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = test(model, test_loader, criterion)
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Después del entrenamiento, guarda el modelo entrenado
model_path = "data/cvae_mnist.pth"
torch.save(model.state_dict(), model_path)

print("Modelo entrenado guardado correctamente.")