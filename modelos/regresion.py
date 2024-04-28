import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Cargando los datos
datos_estadisticas = pd.read_csv('datos/datos_limpios/estadisticas_bayern_limpios.csv')
datos_resultados = pd.read_csv('datos/datos_limpios/resultados_bayern_limpios.csv')

# Asegúrate de que los nombres de columnas son correctos
print("Columnas en datos_estadisticas:", datos_estadisticas.columns.tolist())
print("Columnas en datos_resultados:", datos_resultados.columns.tolist())

# Preprocesamiento simple: llenar valores faltantes y convertir a numéricos
datos_estadisticas.fillna(0, inplace=True)
datos_resultados.fillna(0, inplace=True)
datos_resultados.replace({'Resultado': {'V': 3, 'E': 1, 'D': 0}}, inplace=True)

# Seleccionando características de ejemplo (asegurándose de que sean numéricas)
features = ['Goles a favor', 'Goles en contra', 'xG', 'xGA', 'Posesión']
for feature in features:
    datos_resultados[feature] = pd.to_numeric(datos_resultados[feature], errors='coerce')

# Comprobando por cualquier valor NaN y ver el tamaño de los datos
print("Valores NaN después de la conversión numérica:", datos_resultados[features].isna().sum())
datos_resultados.dropna(inplace=True)  # Limpieza final
print("Tamaño de datos_resultados después de eliminar NaN:", datos_resultados.shape)

# Asegurándose de que aún hay datos para procesar
if datos_resultados.empty:
    raise ValueError("No hay datos para procesar después de la limpieza.")

X = datos_resultados[features]
y = datos_resultados['Resultado']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Predicción en el conjunto de prueba
predicciones = modelo.predict(X_test)

# Calcular la precisión
precision = accuracy_score(y_test, predicciones)
print(f'Precisión del modelo: {precision:.2f}')

# Gráfica de importancia de las características
importancias = modelo.feature_importances_
caracteristicas = X.columns
indices = sorted(range(len(importancias)), key=lambda i: importancias[i], reverse=True)

plt.figure()
plt.title("Importancia de las Características")
plt.bar(range(X.shape[1]), importancias[indices], align="center")
plt.xticks(range(X.shape[1]), [caracteristicas[i] for i in indices], rotation=90)
plt.show()




