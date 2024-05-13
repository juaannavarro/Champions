
import pandas as pd

print("Cargando datos...")
data_23_24 = pd.read_csv('datos/limpios/23_24_limpio.csv')
print("Datos cargados con éxito.")
print(data_23_24.head())

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Ejemplo simplificado de definición del modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Asegúrate de que la dimensión de entrada es correcta
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print("Modelo compilado.")

# Intenta entrenar el modelo con un subconjunto pequeño de los datos
print("Entrenando el modelo con un subconjunto de datos...")
model.fit(X_train[:10], y_train[:10], epochs=1, batch_size=1)
print("Modelo entrenado en subconjunto.")

# Si esto funciona, el problema puede estar relacionado con el tamaño del dataset o la configuración del modelo

print("Entrenando el modelo con todos los datos...")