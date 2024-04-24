import pandas as pd

def analizar_goleadores(ruta):
    """Análisis para obtener el top 10 de goleadores."""
    df = pd.read_csv(ruta)
    resultado = df.nlargest(10, 'Goles')  # Asumiendo que hay una columna 'Goles'
    return resultado

#def analizar_fase_eliminatoria(ruta):
    """Análisis para obtener los equipos que han pasado de ronda, con todas sus estadísticas."""
    df = pd.read_csv(ruta)
    # Asumiendo que hay una columna que indica si el equipo pasó
    resultado = df[df['Paso'] == 'Sí']  # Cambia 'Paso' y 'Sí' según tus datos
    return resultado

def analizar_estadisticas_equipo(ruta):
    """Análisis para obtener los 5 jugadores más importantes de cada equipo."""
    df = pd.read_csv(ruta)
    # Asumiendo que las columnas importantes son 'Minutos' y 'Goles'
    resultado = df.nlargest(5, ['Minutos'])
    return resultado

if __name__ == "__main__":
    # Ejecutar análisis específicos para cada archivo
    goleadores = analizar_goleadores('datos_limpios/goleadores_champions_league_completo_limpios.csv')
    goleadores.to_csv('analisis_resultados/goleadores_top10.csv', index=False)

    #for fase in ['octavos', 'cuartos', 'semis', 'fase_grupos']:
        #resultados_fase = analizar_fase_eliminatoria(f'datos_limpios/{fase}_limpios.csv')
        #resultados_fase.to_csv(f'analisis_resultados/{fase}_resultados.csv', index=False)

    for equipo in ['dormund', 'real_madrid', 'bayern', 'psg']:
        estadisticas_equipo = analizar_estadisticas_equipo(f'datos_limpios/estadisticas_{equipo}_limpios.csv')
        estadisticas_equipo.to_csv(f'analisis_resultados/estadisticas_{equipo}_top5.csv', index=False)
