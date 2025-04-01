import os
from dotenv import load_dotenv
load_dotenv()  # Carrega as variáveis do arquivo .env

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Recuperar variáveis de ambiente
MODEL_PATH = os.getenv("MODEL_PATH", "model")
EPOCHS = int(os.getenv("EPOCHS", 10))

# Cria o diretório para salvar o modelo, se não existir
os.makedirs(MODEL_PATH, exist_ok=True)

# Transformações para os dados (normalização padrão para MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Baixar os datasets de treinamento e teste
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Definição do modelo CNN
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

model = Net()

# Usar apenas CPU
device = torch.device("cpu")
model.to(device)

optimizer = optim.Adam(model.parameters())

# Loop de treinamento
for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

# Salvar o modelo treinado
model_file = os.path.join(MODEL_PATH, "mnist_model.pt")
torch.save(model.state_dict(), model_file)
print("Modelo treinado e salvo com sucesso!")
