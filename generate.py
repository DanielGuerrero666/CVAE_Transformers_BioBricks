import torch
from torchvision.utils import save_image
from models.CVAE import CVAE

# Define el dispositivo a utilizar (GPU si está disponible, de lo contrario CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ruta del modelo
model_path = "data/cvae_mnist.pth"

# Carga el modelo entrenado
model = CVAE(input_dim=784, latent_dim=20, hidden_dim=128, output_dim=784).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Número de imágenes a generar
num_images = 10

# Genera las imágenes
with torch.no_grad():
    for i in range(num_images):
        # Genera muestras aleatorias en el espacio latente
        latent_sample = torch.randn(1, 20).to(device)
        # Decodifica las muestras latentes en imágenes
        generated_image = model.decoder(latent_sample)
        # Guarda las imágenes generadas
        save_image(generated_image.view(1, 1, 28, 28), f"images/generated_image_{i+1}.png")

print(f"{num_images} imágenes generadas exitosamente.")