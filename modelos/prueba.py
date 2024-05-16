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

# Preparación de características
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(data[['Home', 'Away']])  # Ajustar con datos de entrenamiento
features_encoded = encoder.transform(data[['Home', 'Away']])  # Transformar datos de entrenamiento

X = np.hstack((features_encoded, data[['HomeGoals', 'AwayGoals']].values))  # Combinar con otras características
y = data['Result'].values

# Preparación de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo y entrenamiento (como definido anteriormente)

# Preparar datos para predicción
nuevos_partidos = pd.DataFrame({
    'Home': ['RB Leipzigde', 'FC Copenhagendk', 'Paris S-Gfr', 'Lazioit'],
    'Away': ['esReal Madrid', 'engManchester City', 'esReal Sociedad', 'deBayern Munich']
})
features_nuevas = encoder.transform(nuevos_partidos)  # Transformar nuevos datos

# Verificar y ajustar características
print("Características generadas por OneHotEncoder:", encoder.get_feature_names_out())
print("Forma de las características nuevas:", features_nuevas.shape)

# Proceso de predicción (como definido anteriormente)
