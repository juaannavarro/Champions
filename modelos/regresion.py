import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Función para calcular estadísticas clave por posición
def summarize_by_position(data):
    # Convertir columnas de interés a numérico y manejar valores faltantes
    data['Minutos'] = pd.to_numeric(data['Minutos'].str.replace(',', ''), errors='coerce')
    data = data.dropna(subset=['Minutos'])  # Eliminando filas donde los minutos no son disponibles
    
    # Agregando métricas de interés
    metrics = ['Goles', 'Asistencias', 'xG: Goles esperados', 'npxG: Goles esperados (xG) sin contar penaltis', 
               'xAG: Exp. Assisted Goals', 'Minutos', 'Partidos jugados']
    # Agrupando por posición y calculando promedios ponderados por minutos jugados
    summary = data.groupby('Posición')[metrics].apply(
        lambda x: pd.Series({
            'Goles por 90': (x['Goles'] / x['Minutos']).sum() * 90,
            'Asistencias por 90': (x['Asistencias'] / x['Minutos']).sum() * 90,
            'xG por 90': (x['xG: Goles esperados'] / x['Minutos']).sum() * 90,
            'npxG por 90': (x['npxG: Goles esperados (xG) sin contar penaltis'] / x['Minutos']).sum() * 90,
            'xAG por 90': (x['xAG: Exp. Assisted Goals'] / x['Minutos']).sum() * 90,
            'Partidos jugados promedio': x['Partidos jugados'].mean(),
            'Minutos promedio': x['Minutos'].mean()
        })
    ).reset_index()
    return summary

def calculate_points(results_df):
    # Suponiendo que los resultados están en la columna 'Resultado' con V=Victoria, E=Empate, D=Derrota
    # y la competición está en la columna 'Competición' con el valor 'Champions Lg' para la Champions League
    results_df['Punto_Basico'] = results_df['Resultado'].map({'V': 3, 'E': 1, 'D': 0})
    
    # Añadir un plus por jugar en la Champions League
    results_df['Puntos'] = results_df.apply(
        lambda x: x['Punto_Basico'] * 1.5 if x['Competición'] == 'Champions Lg' else x['Punto_Basico'], axis=1
    )
    
    return results_df['Puntos'].sum()
def prepare_data_for_model(team_stats, team_results):
    summary = summarize_by_position(team_stats)
    points = calculate_points(team_results)
    summary['Puntos'] = points
    return summary

def analyze_and_scale_features(df):
    # Visualización de las relaciones y distribuciones
    sns.pairplot(df[['Goles por 90', 'Asistencias por 90', 'xG por 90', 'npxG por 90', 'xAG por 90', 'Puntos']])
    plt.show()

    # Normalización de los datos
    scaler = StandardScaler()
    feature_columns = ['Goles por 90', 'Asistencias por 90', 'xG por 90', 'npxG por 90', 'xAG por 90']
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

def train_and_evaluate_model(df):
    # Seleccionar las columnas relevantes basadas en tus datos
    X = df[['Goles por 90', 'Asistencias por 90', 'xG por 90']]
    y = df['Puntos']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predecir los puntos en el conjunto de prueba y calcular R^2
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mean_squared_error(y_test, y_pred)}, R^2: {r2:.2f}')

    # Visualizar resultados reales vs. predicciones
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    plt.xlabel('Actual Points')
    plt.ylabel('Predicted Points')
    plt.title('Actual vs. Predicted Points')
    plt.show()

    return model

def main():
    # Cargar y preparar datos
    teams = ['bayern', 'psg', 'dortmund', 'real_madrid']
    all_data = pd.DataFrame()

    for team in teams:
        stats = pd.read_csv(f'datos/estadisticas_{team}.csv')
        results = pd.read_csv(f'datos/resultados_{team}.csv')
        team_data = prepare_data_for_model(stats, results)
        all_data = pd.concat([all_data, team_data], ignore_index=True)

    # Entrenar el modelo y visualizar resultados
    model = train_and_evaluate_model(all_data)
    print('Model trained successfully.')


if __name__ == '__main__':
    main()








