import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Cargar datos originales
data = pd.read_csv('datos/limpios/equipos_limpio.csv')

# Limpiar y preparar el objetivo: resultado del partido ajustado para que comience en 0
data['Score'] = data['Score'].str.strip()  # Limpia espacios en blanco
data = data[data['Score'].str.contains('^[0-9]+[–-][0-9]+$')]  # Asegura formato correcto
data['HomeGoals'], data['AwayGoals'] = data['Score'].str.split('–', expand=True).astype(int).values.T
data['Result'] = np.where(data['HomeGoals'] > data['AwayGoals'], 2, np.where(data['HomeGoals'] < data['AwayGoals'], 0, 1))

# Preparar características
features = data[['Home', 'Away']]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Ignorar categorías desconocidas
features_encoded = encoder.fit_transform(features)
X = features_encoded
y = data['Result'].values

# Cargar el nuevo archivo CSV
data_new = pd.read_csv('datos/limpios/partidos23_24_limpio.csv')

# Limpiar y preparar el objetivo para el nuevo conjunto de datos
data_new['Score'] = data_new['Score'].str.strip()  # Limpia espacios en blanco
data_new = data_new[data_new['Score'].str.contains('^[0-9]+[-][0-9]+$')]  # Asegura formato correcto
data_new['HomeGoals'], data_new['AwayGoals'] = data_new['Score'].str.split('-', expand=True).astype(int).values.T
data_new['Result'] = np.where(data_new['HomeGoals'] > data_new['AwayGoals'], 2, np.where(data_new['HomeGoals'] < data_new['AwayGoals'], 0, 1))

# Preparar características usando el mismo OneHotEncoder
features_new = data_new[['Home', 'Away']]
features_encoded_new = encoder.transform(features_new)
X_new = features_encoded_new
y_new = data_new['Result'].values

# Combinar los nuevos datos codificados con los existentes
X_combined = np.vstack([X, X_new])
y_combined = np.concatenate([y, y_new])

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Escalado de características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir a tensores de PyTorch
X_train_tensor = torch.tensor(X_train.astype(np.float32))
y_train_tensor = torch.tensor(y_train.astype(np.int64))  # Asegurarse que sea int64 para CrossEntropyLoss
X_test_tensor = torch.tensor(X_test.astype(np.float32))
y_test_tensor = torch.tensor(y_test.astype(np.int64))

# Crear Dataset y DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Definir el modelo de red neuronal mejorado
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 128)  # Capa ampliada
        self.dropout1 = nn.Dropout(0.5)  # Dropout para regularización
        self.layer2 = nn.Linear(128, 128)  # Segunda capa con más neuronas
        self.dropout2 = nn.Dropout(0.5)  # Dropout adicional
        self.layer3 = nn.Linear(128, 64)  # Tercera capa
        self.dropout3 = nn.Dropout(0.5)  # Dropout
        self.output = nn.Linear(64, 3)  # 3 clases para victoria, empate y derrota

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        x = self.output(x)
        return x

model = NeuralNet()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loop de entrenamiento
for epoch in range(10):  # Número de épocas
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluación del modelo
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
# Usar un optimizador diferente o ajustar la tasa de aprendizaje
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Cambio a SGD con momentum

# Añadir regularización L2
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Añadir weight_decay para L2

print(f'Accuracy: {100 * correct / total}%')





