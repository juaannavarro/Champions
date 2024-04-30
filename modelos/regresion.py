import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('datos/zz.csv')

df = df[df['Score'].str.contains('–')]  # Asumiendo que el guión '–' es el delimitador de goles

# Extraer goles de la columna 'Score'
df['Home_Goals'] = df['Score'].str.split('–').str[0].astype(int)
df['Away_Goals'] = df['Score'].str.split('–').str[1].astype(int)

# Filtrar por las últimas tres temporadas
seasons_of_interest = ['2021-2022', '2022-2023']
df = df[df['Season'].isin(seasons_of_interest)]

# Asignar puntos
df['Home_Points'] = (df['Home_Goals'] > df['Away_Goals']) * 3 + (df['Home_Goals'] == df['Away_Goals']) * 1

# Preparar variables para la regresión
X = df[['Home_Goals', 'Away_Goals']]  # Puedes añadir más variables predictoras
y = df['Home_Points']  # Predecir los puntos del equipo en casa

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)
df['Predicted_Points'] = model.predict(df[['Home_Goals', 'Away_Goals']])

# Predecir los resultados para el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f'MSE: {mse}')
print(f'R²: {r2}')

# Función para graficar los resultados de cada temporada
def plot_season_data(season):
    season_data = df[df['Season'] == season]
    plt.figure(figsize=(10, 6))
    plt.scatter(season_data['Home_Points'], season_data['Predicted_Points'], alpha=0.5)
    plt.title(f'Real vs Predicción - Temporada {season}')
    plt.xlabel('Puntos Reales del Equipo en Casa')
    plt.ylabel('Puntos Predichos del Equipo en Casa')
    plt.plot([season_data['Home_Points'].min(), season_data['Home_Points'].max()], [season_data['Home_Points'].min(), season_data['Home_Points'].max()], 'k--')
    plt.grid(True)
    plt.show()

# Graficar para cada temporada de interés
for season in seasons_of_interest:
    plot_season_data(season)

 
print(df['Season'].value_counts())










