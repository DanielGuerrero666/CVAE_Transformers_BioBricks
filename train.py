import torch
import torch.nn as nn
import torch.optim as optim
from data.MNIST_dataset import train_loader, test_loader  # Importa los dataloaders de tu archivo MNIST_dataset.py
from models.CVAE import CVAE  # Importa el modelo CVAE de tu archivo CVAE.py
import matplotlib.pyplot as plt

# Define el dispositivo a utilizar (GPU si está disponible, de lo contrario CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Listas para ploteo posterior
train_losses = []
test_losses = []

# Hiperparámetros
input_dim = 784  # Dimensión de entrada de las imágenes MNIST (28x28 = 784)
latent_dim = 20  # Dimensión del espacio latente
output_dim = input_dim  # Dimensión de salida (igual a la dimensión de entrada)
hidden_dim = 128  # Dimensión de las capas ocultas (Neuronas en cada capa oculta)
learning_rate = 0.001 # Tasa de aprendizaje
num_epochs = 10

# Inicializa el modelo
model = CVAE(input_dim, latent_dim, hidden_dim, output_dim).to(device)

# Define la función de pérdida (MSELoss para la reconstrucción)
reconstruction_loss = nn.MSELoss()

# Define la función de pérdida de KL Divergence
def kl_divergence(mu, logvar):
    # Calcula la pérdida de KL para cada muestra en el batch
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss

# Define el optimizador (Adam)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Función de entrenamiento
def train(model, train_loader, optimizer, reconstruction_loss):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)  # Aplana las imágenes
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = reconstruction_loss(recon_batch, data) + kl_divergence(mu, logvar)  # Calcula la pérdida total
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    return running_loss / len(train_loader.dataset)

# Función de evaluación
def test(model, test_loader, reconstruction_loss):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, 784).to(device)  # Aplana las imágenes
            recon_batch, mu, logvar = model(data)
            test_loss += (reconstruction_loss(recon_batch, data) + kl_divergence(mu, logvar)).item()  # Suma la pérdida total
    test_loss /= len(test_loader.dataset)
    return test_loss

# Entrenamiento del modelo
for epoch in range(1, num_epochs + 1):
    train_loss = train(model, train_loader, optimizer, reconstruction_loss)
    test_loss = test(model, test_loader, reconstruction_loss)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Después del entrenamiento, guarda el modelo entrenado
model_path = "data/cvae_mnist.pth"
torch.save(model.state_dict(), model_path)

print("Modelo entrenado guardado correctamente.")

# Plotting the learning curve
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()