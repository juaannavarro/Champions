import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Función para limpiar y preparar los datos
def preparar_datos(data):
    data = data.copy()
    data['Score'] = data['Score'].str.replace('–', '-').str.replace('—', '-').str.strip()
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

# Configuración de OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(data_historica[['Home', 'Away']])
X_actual = encoder.transform(data_actual[['Home', 'Away']])
X_historica = encoder.transform(data_historica[['Home', 'Away']])
y_historica = data_historica['Result'].values

# Tensor de datos
X_train_tensor = torch.tensor(X_historica.astype(np.float32))
y_train_tensor = torch.tensor(y_historica.astype(np.int64))
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Definición del modelo neuronal
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(X_historica.shape[1], 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 3)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
for epoch in range(10):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Simulación de resultados
X_actual_tensor = torch.tensor(X_actual.astype(np.float32))
with torch.no_grad():
    predictions = model(X_actual_tensor)
    predicted_results = torch.argmax(predictions, dim=1)
data_actual['PredictedResult'] = predicted_results.numpy()

# Función para simular cada ronda
def simular_ronda(ronda):
    resultados = ronda[['Home', 'Away', 'PredictedResult']]
    for index, match in resultados.iterrows():
        if match['PredictedResult'] == 2:
            print(f"{match['Home']} vence a {match['Away']}")
        elif match['PredictedResult'] == 0:
            print(f"{match['Away']} vence a {match['Home']}")
        else:
            print(f"{match['Home']} y {match['Away']} empatan")

# Simulación por ronda
for ronda in ['Round of 16', 'Quarter-finals', 'Semi-finals', 'Final']:
    print(f"\nResultados {ronda}:")
    simular_ronda(data_actual[data_actual['Round'] == ronda])

# Filtrar los partidos de la final
final_partidos = data_actual[data_actual['Round'] == 'Final']

# Verificar si hay partidos en la final
if final_partidos.empty:
    print("No hay partidos etiquetados como 'Final' en los datos.")
else:
    # Transformar los datos de la final para la predicción
    X_final = encoder.transform(final_partidos[['Home', 'Away']])
    X_final_tensor = torch.tensor(X_final.astype(np.float32))

    # Hacer la predicción para la final
    with torch.no_grad():
        predictions_final = model(X_final_tensor)
        predicted_results_final = torch.argmax(predictions_final, dim=1)
        probabilities_final = torch.softmax(predictions_final, dim=1)

    # Mostrar los resultados predichos para la final
    final_partidos['PredictedResult'] = predicted_results_final.numpy()
    final_partidos['Probability'] = probabilities_final.numpy().max(axis=1)
    for index, row in final_partidos.iterrows():
        home_team = row['Home']
        away_team = row['Away']
        predicted_result = row['PredictedResult']
        probability = row['Probability']
        result_string = f"empate" if predicted_result == 1 else (f"victoria para {home_team}" if predicted_result == 2 else f"victoria para {away_team}")
        print(f"Final: {home_team} vs {away_team} - Resultado predicho: {result_string} con probabilidad de {probability:.2f}")


