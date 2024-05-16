import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Carga de datos y limpieza inicial
data = pd.read_csv('datos/limpios/equipos_limpio.csv')
data['Score'] = data['Score'].str.replace('–', '-').str.strip()
data = data[data['Score'].str.contains('^\d+-\d+$')]
data[['HomeGoals', 'AwayGoals']] = data['Score'].str.split('-', expand=True).astype(int)
data['Result'] = np.where(data['HomeGoals'] > data['AwayGoals'], 2,
                          np.where(data['HomeGoals'] < data['AwayGoals'], 0, 1))

# Cargar datos adicionales y realizar fusiones simplificadas
features = data[['Home', 'Away', 'HomeGoals', 'AwayGoals']]  # Considerar incluir solo metas

# Preparación de características
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
features_encoded = encoder.fit_transform(features[['Home', 'Away']])  # Codificar solo equipos
X = np.hstack((features_encoded, features[['HomeGoals', 'AwayGoals']].values))  # Añadir goles directamente
y = data['Result'].values

# Preparación de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo neuronal simplificado
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 64)  # Reducir tamaño y complejidad
        self.layer2 = nn.Linear(64, 3)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Configuración de entrenamiento
model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for inputs, labels in DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()), batch_size=10, shuffle=True):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluación
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()), batch_size=10, shuffle=False):
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {100 * correct / total}%')



