import os
from dotenv import load_dotenv
load_dotenv()  # Carrega as variáveis do arquivo .env

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recuperar variáveis de ambiente
MODEL_PATH = os.getenv("MODEL_PATH", "model")
IMAGE_PATH = os.getenv("IMAGE_PATH", "numero.png")
model_file = os.path.join(MODEL_PATH, "mnist_model.pt")

# Definição do mesmo modelo utilizado no treinamento
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def predict_digit(image_path):
    # Carregar o modelo treinado
    model = Net()
    model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
    model.eval()

    # Carregar a imagem em escala de cinza
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Imagem não encontrada!")
        return None

    # Redimensionar para 28x28 pixels
    image = cv2.resize(image, (28, 28))
    # Inverter as cores se necessário (se o fundo for branco)
    image = 255 - image
    # Normalizar a imagem (mesmo procedimento da normalização do MNIST)
    image = image.astype("float32") / 255.0
    image = (image - 0.1307) / 0.3081  # Normalização padronizada
    # Ajustar as dimensões para (1, 1, 28, 28)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    # Converter para tensor
    tensor_image = torch.tensor(image)

    # Previsão
    with torch.no_grad():
        output = model(tensor_image)
        pred = output.argmax(dim=1, keepdim=True).item()
        confidence = torch.exp(output.max()).item() * 100

    # Exibir a imagem e a predição
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Predição: {pred} ({confidence:.2f}%)")
    plt.show()

    return pred

if __name__ == "__main__":
    prediction = predict_digit(IMAGE_PATH)
    if prediction is not None:
        print(f"O número previsto é: {prediction}")
