import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar datos
df = pd.read_csv('datos/limpios/ptos_equipo_limpio.csv')

# Asegurarse de que 'Pts' es numérico
df['Pts'] = pd.to_numeric(df['Pts'], errors='coerce')

# Eliminar filas donde 'Pts' es NaN después de intentar la conversión
df = df.dropna(subset=['Pts'])

# Asegurarse de que otras columnas usadas como características también son numéricas
for col in ['GF', 'GA', 'GD']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Tratar NaNs después de conversión
df.fillna(df.median(), inplace=True)

# Seleccionar características y variable objetivo
X = df[['GF', 'GA', 'GD']]  # Características
y = df['Pts']  # Variable objetivo

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')




print(df.dtypes)
print(df.head())


import matplotlib.pyplot as plt

# Calcula residuos
residuos = y_test - y_pred

# Gráfico de dispersión de residuos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuos)
plt.axhline(0, color='red', linestyle='--')
plt.title('Gráfico de Residuos')
plt.xlabel('Valores Reales de Puntos')
plt.ylabel('Residuos')
plt.show()

# Histograma de residuos
plt.figure(figsize=(10, 6))
plt.hist(residuos, bins=30, edgecolor='black')
plt.title('Histograma de Residuos')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.show()
    


archivo_temporada_actual = 'datos/limpios/23_24_limpio.csv'
df_actual = pd.read_csv(archivo_temporada_actual)

# Asegúrate de que las columnas utilizadas en el modelo estén presentes y correctamente formateadas
df_actual['GF'] = pd.to_numeric(df_actual['GF'], errors='coerce')
df_actual['GA'] = pd.to_numeric(df_actual['GA'], errors='coerce')
df_actual['GD'] = pd.to_numeric(df_actual['GD'], errors='coerce')

# Llena los posibles valores nulos que podrían haber surgido durante la conversión
df_actual.fillna(df_actual.median(), inplace=True)

# Predicción para la nueva temporada
X_nueva_temporada = df_actual[['GF', 'GA', 'GD']]
predicciones_pts = model.predict(X_nueva_temporada)

# Añadir las predicciones al DataFrame para una mejor visualización
df_actual['Predicted_Pts'] = predicciones_pts


# Visualización de los puntos predichos
plt.figure(figsize=(14, 7))
plt.bar(df_actual['Squad'], df_actual['Predicted_Pts'], color='skyblue')
plt.xlabel('Equipos')
plt.ylabel('Puntos Predichos')
plt.title('Predicciones de Puntos para la Temporada Actual')
plt.xticks(rotation=90)
plt.show()


# Supongamos que df_actual es tu DataFrame y ya contiene las predicciones en 'Predicted_Pts'
# Asegúrate de que los datos estén listos para mostrar
df_actual['Predicted_Pts'] = pd.to_numeric(df_actual['Predicted_Pts'], errors='coerce')
df_actual['Squad'] = df_actual['Squad'].astype(str)

# Ordenar los equipos por puntos predichos, de mayor a menor
df_sorted = df_actual.sort_values(by='Predicted_Pts', ascending=False)

# Selecciona las columnas que quieres mostrar
columns_to_display = ['Squad', 'Predicted_Pts']

# Muestra la tabla
print(df_sorted[columns_to_display])

print('EQUIPOS CLASIFICADOS PARA LA CHAMPIONS LEAGUE')

# Suponiendo que df_sorted ya contiene tu DataFrame con 'Squad' y 'Predicted_Pts'
equipos_seleccionados = ['esReal Madrid', 'deBayern Munich', 'frParis S-G', 'deDortmund']

# Filtrar el DataFrame para incluir solo los equipos especificados
df_filtered = df_sorted[df_sorted['Squad'].isin(equipos_seleccionados)]
# Mostrar la tabla filtrada
print(df_filtered[['Squad', 'Predicted_Pts']])
