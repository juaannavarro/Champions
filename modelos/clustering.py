import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
print('• Aprendizaje No-Supervisado en Python')
# Cargar los datos
df = pd.read_csv('datos/limpios/ptos_equipo_limpio.csv')

# Convertir columnas a numérico y manejar errores
for column in ['GF', 'GA', 'GD', 'Pts']:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Lidiar con valores NaN que podrían haber surgido durante la conversión
df.fillna(df.median(), inplace=True)

# Seleccionar características para el clustering
features = df[['GF', 'GA', 'GD', 'Pts']]

# Estandarizar las características
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Aplicar K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Añadir la asignación de clusters al DataFrame original
df['Cluster'] = clusters

# Mostrar las estadísticas de cada cluster
print(df.groupby('Cluster').mean())

# Visualizar los resultados (opcional)
import matplotlib.pyplot as plt

plt.scatter(df['GD'], df['Pts'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Diferencia de Goles')
plt.ylabel('Puntos')
plt.title('Clustering de Equipos por Diferencia de Goles y Puntos')
plt.colorbar(label='Cluster')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generar datos sintéticos
np.random.seed(42)
tiempo_en_pagina = np.random.normal(loc=5, scale=2, size=300)
total_clicks = tiempo_en_pagina + np.random.normal(loc=0, scale=1, size=300)

X = np.column_stack((tiempo_en_pagina, total_clicks))

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
centros = kmeans.cluster_centers_
plt.scatter(centros[:, 0], centros[:, 1], c='red', s=250, alpha=0.75, marker='x')
plt.title('Clustering con K-Means')
plt.xlabel('Tiempo en la página (minutos)')
plt.ylabel('Total de clicks')
plt.colorbar(label='Cluster ID')
plt.show()

print('• Ingeniería de variables (Feature engineering)')




import pandas as pd

# Cargar los datos
df = pd.read_csv('datos/limpios/jugadores_limpio.csv')

# Verificar las columnas disponibles
print("Columnas disponibles en el DataFrame:", df.columns)

# Convertir todas las columnas necesarias a numérico, asegurando de manejar correctamente los errores
# Asumiendo que 'Goles' es el nombre correcto según tu último mensaje
cols_to_convert = ['PJ', 'Titular', 'Goles', 'Asistencias']
if 'Tiros' in df.columns:
    cols_to_convert.append('Tiros')

for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Llenar los valores NaN resultantes de la conversión fallida con 0 (o otro valor que consideres apropiado)
df[cols_to_convert] = df[cols_to_convert].fillna(0)

# Crear nuevas variables solo si la columna 'Tiros' existe
if 'Tiros' in df.columns:
    df['Goles_por_Tiro'] = df['Goles'] / df['Tiros'] if df['Tiros'].sum() != 0 else 0
else:
    print("La columna 'Tiros' no existe en el DataFrame.")
df['Porcentaje_Titular'] = (df['Titular'] / df['PJ']) * 100
df['Asistencias_por_Partido'] = df['Asistencias'] / df['PJ']

# Mostrar las nuevas columnas y verificar su correcta creación
if 'Tiros' in df.columns:
    print(df[['Goles_por_Tiro', 'Porcentaje_Titular', 'Asistencias_por_Partido']].head())
else:
    print(df[['Porcentaje_Titular', 'Asistencias_por_Partido']].head())



