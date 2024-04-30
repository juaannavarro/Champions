import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
# Cargar los datos
archivo = 'datos/pts_equipos.csv'
df = pd.read_csv(archivo)

#eliminar notas
df.drop('Notes', axis=1, inplace=True)

# Eliminar las filas que solo contienen comas en su mayoría
df.dropna(how='all', inplace=True)  # Elimina filas que son totalmente vacías
df = df.dropna(subset=[col for col in df.columns if col != 'Season'], how='all')  # Elimina filas que solo tienen 'Season'

# Suponiendo que 'Top Team Scorer' es el nombre de la columna y los datos tienen el formato 'Jugador (Goles)'
if 'Top Team Scorer' in df.columns:
    # Separar el nombre del goleador y los goles en dos columnas nuevas
    df['Scorer'] = df['Top Team Scorer'].apply(lambda x: re.findall(r"([a-zA-Z\s]+)", x)[0].strip() if pd.notna(x) else x)
    df['Goals'] = df['Top Team Scorer'].apply(lambda x: int(re.findall(r"(\d+)", x)[0]) if pd.notna(x) and re.search(r"(\d+)", x) else x)

    # Eliminar la columna original 'Top Team Scorer'
    df.drop('Top Team Scorer', axis=1, inplace=True)
# Imputar valores nulos
df['Scorer'].fillna('Desconocido', inplace=True)  # Imputar con 'Desconocido' para los goleadores
df['Goals'].fillna(0, inplace=True)  # Imputar con 0 para los goles
print(df.head())

#eliminar attendance
df.drop('Attendance', axis=1, inplace=True)
print(df.info())
# Mostrar la cantidad de valores nulos por columna
print("Cantidad de valores nulos por columna:")
print(df.isnull().sum())

# Opcional: Mostrar el porcentaje de valores nulos por columna
print("\nPorcentaje de valores nulos por columna:")
print(df.isnull().mean() * 100)


# Guardar los datos limpios en un nuevo archivo CSV

archivo_limpio = 'datos/limpios/ptos_equipo_limpio.csv'
df.to_csv(archivo_limpio, index=False)