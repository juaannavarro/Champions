import pandas as pd

def cargar_datos(ruta):
    """Carga datos desde un archivo CSV a un DataFrame."""
    return pd.read_csv(ruta)

def limpiar_datos(df):
    """Realiza operaciones de limpieza en el DataFrame."""
    df = df.dropna()  # Eliminar filas con valores NaN
    df = df.drop_duplicates()  # Eliminar duplicados
    return df

if __name__ == "__main__":
    # Lista de nombres de archivos CSV a procesar
    nombres_archivos = ['goleadores_champions_league_completo.csv', 'octavos.csv', 'cuartos.csv',
                        'semis.csv', 'clasificacion_fase_grupos_champions_league.csv', 
                        'resultados_dormund.csv', 'resultados_real_madrid.csv', 
                        'resultados_bayern.csv', 'resultados_psg.csv', 'estadisticas_real_madrid.csv',
                        'estadisticas_bayern.csv', 'estadisticas_psg.csv', 'estadisticas_dormund.csv']
    
    # Procesar cada archivo en la lista
    for nombre in nombres_archivos:
        ruta = f'datos/{nombre}'  # Construye la ruta completa al archivo
        datos = cargar_datos(ruta)
        datos_limpios = limpiar_datos(datos)
        
        # Construir el nombre del archivo de salida
        nombre_limpios = nombre.replace('.csv', '_limpios.csv')
        ruta_salida = f'datos/datos_limpios/{nombre_limpios}'
        
        # Guardar los datos limpios
        datos_limpios.to_csv(ruta_salida, index=False)
        print(f"Datos limpios guardados en {ruta_salida}")

