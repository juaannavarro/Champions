import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Función para limpiar y preparar los datos
def preparar_datos(data):
    data = data.copy()
    data['Score'] = data['Score'].str.replace('–', '-').str.strip()
    data = data[data['Score'].str.contains('^\d+-\d+$')]
    scores = data['Score'].str.split('-', expand=True)
    data['HomeGoals'] = pd.to_numeric(scores[0], errors='coerce')
    data['AwayGoals'] = pd.to_numeric(scores[1], errors='coerce')
    data.dropna(subset=['HomeGoals', 'AwayGoals'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['HomeGoals'] = data['HomeGoals'].astype(int)
    data['AwayGoals'] = data['AwayGoals'].astype(int)
    data['Result'] = np.where(data['HomeGoals'] > data['AwayGoals'], 2,
                              np.where(data['HomeGoals'] < data['AwayGoals'], 0, 1))
    return data

# Carga y preparación de datos
data_historica = pd.read_csv('datos/limpios/equipos_limpio.csv')
data_actual = pd.read_csv('datos/limpios/partidos23_24_limpio.csv')
data_historica = preparar_datos(data_historica)
data_actual = preparar_datos(data_actual)

# Preparación de características para ambos modelos
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(data_historica[['Home', 'Away']])

# Primer modelo: incluye 'HomeGoals' y 'AwayGoals'
X_historica1 = encoder.transform(data_historica[['Home', 'Away']])
X_historica1 = np.hstack((X_historica1, data_historica[['HomeGoals', 'AwayGoals']].values))
y_historica = data_historica['Result'].values

# Segundo modelo: sólo equipos
X_historica2 = encoder.transform(data_historica[['Home', 'Away']])

# División de datos y escalado para el primer modelo
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_historica1, y_historica, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train1)
X_test1 = scaler.transform(X_test1)

# División de datos para el segundo modelo
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_historica2, y_historica, test_size=0.2, random_state=42)

# Convertir a tensores para ambos modelos
X_train_tensor1 = torch.tensor(X_train1.astype(np.float32))
y_train_tensor1 = torch.tensor(y_train1.astype(np.int64))
X_test_tensor1 = torch.tensor(X_test1.astype(np.float32))
y_test_tensor1 = torch.tensor(y_test1.astype(np.int64))

X_train_tensor2 = torch.tensor(X_train2.astype(np.float32))
y_train_tensor2 = torch.tensor(y_train2.astype(np.int64))
X_test_tensor2 = torch.tensor(X_test2.astype(np.float32))
y_test_tensor2 = torch.tensor(y_test2.astype(np.int64))

train_dataset1 = TensorDataset(X_train_tensor1, y_train_tensor1)
train_dataset2 = TensorDataset(X_train_tensor2, y_train_tensor2)
train_loader1 = DataLoader(train_dataset1, batch_size=10, shuffle=True)
train_loader2 = DataLoader(train_dataset2, batch_size=10, shuffle=True)

# Definición del primer modelo neuronal
class NeuralNet1(nn.Module):
    def __init__(self):
        super(NeuralNet1, self).__init__()
        self.layer1 = nn.Linear(X_train1.shape[1], 64)
        self.layer2 = nn.Linear(64, 3)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Definición del segundo modelo neuronal
class NeuralNet2(nn.Module):
    def __init__(self):
        super(NeuralNet2, self).__init__()
        self.layer1 = nn.Linear(X_historica2.shape[1], 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 3)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model1 = NeuralNet1()
model2 = NeuralNet2()
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

# Entrenamiento del primer modelo
for epoch in range(10):
    for inputs, labels in train_loader1:
        optimizer1.zero_grad()
        outputs = model1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer1.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluación del primer modelo
model1.eval()
correct1, total1 = 0, 0
with torch.no_grad():
    for inputs, labels in DataLoader(TensorDataset(X_test_tensor1, y_test_tensor1), batch_size=10, shuffle=False):
        outputs = model1(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total1 += labels.size(0)
        correct1 += (predicted == labels).sum().item()
print(f'Accuracy del modelo 1: {100 * correct1 / total1}%')

# Entrenamiento del segundo modelo
for epoch in range(10):
    for inputs, labels in train_loader2:
        optimizer2.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluación del segundo modelo
model2.eval()
correct2, total2 = 0, 0
with torch.no_grad():
    for inputs, labels in DataLoader(TensorDataset(X_test_tensor2, y_test_tensor2), batch_size=10, shuffle=False):
        outputs = model2(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total2 += labels.size(0)
        correct2 += (predicted == labels).sum().item()
print(f'Accuracy del modelo 2: {100 * correct2 / total2}%')

# Preparación de datos actuales para predicción
X_actual1 = encoder.transform(data_actual[['Home', 'Away']])
X_actual1 = np.hstack((X_actual1, data_actual[['HomeGoals', 'AwayGoals']].values))
X_actual2 = encoder.transform(data_actual[['Home', 'Away']])
X_actual_tensor1 = torch.tensor(X_actual1.astype(np.float32))
X_actual_tensor2 = torch.tensor(X_actual2.astype(np.float32))

# Combinación de predicciones
with torch.no_grad():
    pred1_probs = torch.softmax(model1(X_actual_tensor1), dim=1)
    pred2_probs = torch.softmax(model2(X_actual_tensor2), dim=1)
    combined_probs = (pred1_probs + pred2_probs) / 2
    combined_pred = torch.argmax(combined_probs, dim=1)
    combined_max_probs = combined_probs.max(dim=1)[0]

data_actual['PredictedResult'] = combined_pred.numpy()
data_actual['Probability'] = combined_max_probs.numpy()

# Función para simular cada ronda
def simular_ronda(ronda):
    resultados = ronda[['Home', 'Away', 'PredictedResult', 'Probability']]
    for index, match in resultados.iterrows():
        if match['PredictedResult'] == 2:
            print(f"{match['Home']} vence a {match['Away']} con una probabilidad de {match['Probability']:.2f}")
        elif match['PredictedResult'] == 0:
            print(f"{match['Away']} vence a {match['Home']} con una probabilidad de {match['Probability']:.2f}")
        else:
            print(f"{match['Home']} y {match['Away']} empatan con una probabilidad de {match['Probability']:.2f}")

# Simulación por ronda
for ronda in ['Round of 16', 'Quarter-finals', 'Semi-finals', 'Final']:
    print(f"\nResultados {ronda}:")
    simular_ronda(data_actual[data_actual['Round'] == ronda])

# Simular la final entre Real Madrid y Dortmund
final_partidos = pd.DataFrame({
    'Home': ['esReal Madrid'],
    'Away': ['deDortmund']
})

# Transformar los datos de la final para la predicción
X_final1 = encoder.transform(final_partidos[['Home', 'Away']])
X_final1 = np.hstack((X_final1, np.array([[0, 0]])))  # Añadir goles ficticios para mantener la consistencia de las dimensiones
X_final2 = encoder.transform(final_partidos[['Home', 'Away']])
X_final_tensor1 = torch.tensor(X_final1.astype(np.float32))
X_final_tensor2 = torch.tensor(X_final2.astype(np.float32))

# Hacer la predicción para la final
with torch.no_grad():
    pred_final1_probs = torch.softmax(model1(X_final_tensor1), dim=1)
    pred_final2_probs = torch.softmax(model2(X_final_tensor2), dim=1)
    combined_final_probs = (pred_final1_probs + pred_final2_probs) / 2
    predicted_final_result = torch.argmax(combined_final_probs, dim=1)
    probabilities_final = combined_final_probs.max(dim=1)[0]

# Mostrar los resultados predichos para la final
final_partidos['PredictedResult'] = predicted_final_result.numpy()
final_partidos['Probability'] = probabilities_final.numpy()
for index, row in final_partidos.iterrows():
    home_team = row['Home']
    away_team = row['Away']
    predicted_result = row['PredictedResult']
    probability = row['Probability']
    result_string = f"empate" if predicted_result == 1 else (f"victoria para {home_team}" if predicted_result == 2 else f"victoria para {away_team}")
    print(f"Final: {home_team} vs {away_team} - Resultado predicho: {result_string} con probabilidad de {probability:.2f}")






