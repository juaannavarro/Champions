import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Cargar todos los conjuntos de datos
data = pd.read_csv('datos/limpios/equipos_limpio.csv')

# Limpieza inicial de datos para evitar problemas
data['Score'] = data['Score'].str.replace('–', '-').str.strip()  # Asegúrate de que usas el guión correcto

# Eliminar filas que no contengan el formato esperado de Score 'X-Y'
data = data[data['Score'].str.contains('^\d+-\d+$')]

# Dividir los goles y convertirlos a enteros
data[['HomeGoals', 'AwayGoals']] = data['Score'].str.split('-', expand=True).astype(int)
data['Result'] = np.where(data['HomeGoals'] > data['AwayGoals'], 2,
                          np.where(data['HomeGoals'] < data['AwayGoals'], 0, 1))

# Carga de datos adicionales
goleadores_data = pd.read_csv('datos/limpios/goleadores_limpio.csv')
jugadores_data = pd.read_csv('datos/limpios/jugadores_limpio.csv')
ptos_equipo_data = pd.read_csv('datos/limpios/ptos_equipo_limpio.csv')

# Renombres para asegurar consistencia
goleadores_data.rename(columns={'Equipo': 'Team'}, inplace=True)
jugadores_data.rename(columns={'Equipo': 'Team'}, inplace=True)
ptos_equipo_data.rename(columns={'Squad': 'Team'}, inplace=True)

# Agregaciones
goleadores_aggregated = goleadores_data.groupby('Team').agg({'Goles': 'sum', 'Asistencias': 'sum'}).reset_index()
jugadores_aggregated = jugadores_data.groupby('Team').agg({'Edad': 'mean', 'PJ': 'sum', 'G+A90': 'mean'}).reset_index()

# Fusiones
features = data[['Home', 'Away']]
for df in [goleadores_aggregated, jugadores_aggregated, ptos_equipo_data]:
    features = features.merge(df, how='left', left_on='Home', right_on='Team')
    features.drop(columns='Team', inplace=True)

# Preparación de características
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
features_encoded = encoder.fit_transform(features)
X = features_encoded
y = data['Result'].values

# Preparación de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir a tensores
X_train_tensor = torch.tensor(X_train.astype(np.float32))
y_train_tensor = torch.tensor(y_train.astype(np.int64))
X_test_tensor = torch.tensor(X_test.astype(np.float32))
y_test_tensor = torch.tensor(y_test.astype(np.int64))

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Modelo neuronal
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 128)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.output = nn.Linear(64, 3)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        x = self.output(x)
        return x

# Entrenamiento
model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluación
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {100 * correct / total}%')





