import pandas as pd

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
    points = results_df['Resultado'].map({'V': 3, 'E': 1, 'D': 0}).sum()
    return points

def main():
    # Cargando estadísticas y resultados de cada equipo
    bayern_stats = pd.read_csv('datos/estadisticas_bayern.csv')
    psg_stats = pd.read_csv('datos/estadisticas_psg.csv')
    dortmund_stats = pd.read_csv('datos/estadisticas_dortmund.csv')
    real_madrid_stats = pd.read_csv('datos/estadisticas_real_madrid.csv')
    
    bayern_results = pd.read_csv('datos/resultados_bayern.csv')
    psg_results = pd.read_csv('datos/resultados_psg.csv')
    dortmund_results = pd.read_csv('datos/Resultados_dortmund.csv')
    real_madrid_results = pd.read_csv('datos/resultados_real_madrid.csv')
    
    # Calculando resumen y puntos para cada equipo
    bayern_summary = summarize_by_position(bayern_stats)
    psg_summary = summarize_by_position(psg_stats)
    dortmund_summary = summarize_by_position(dortmund_stats)
    real_madrid_summary = summarize_by_position(real_madrid_stats)

    bayern_points = calculate_points(bayern_results)
    psg_points = calculate_points(psg_results)
    dortmund_points = calculate_points(dortmund_results)
    real_madrid_points = calculate_points(real_madrid_results)
    
    # Imprimiendo resumen y puntos
    print('Bayern de Múnich:')
    print(bayern_summary)
    print('Puntos:', bayern_points)
    print('\nPSG:')
    print(psg_summary)
    print('Puntos:', psg_points)
    print('\nDortmund:')
    print(dortmund_summary)
    print('Puntos:', dortmund_points)
    print('\nReal Madrid:')
    print(real_madrid_summary)
    print('Puntos:', real_madrid_points)

if __name__ == '__main__':
    main()








