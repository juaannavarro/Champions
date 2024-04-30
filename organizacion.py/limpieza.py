import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
# Cargar los datos
archivo = 'datos/23_24.csv'
df = pd.read_csv(archivo)

# Eliminar columna 'Notes'
df.drop('Notes', axis=1, inplace=True)

# Eliminar filas que son completamente vacías o solo contienen 'Season'
df.dropna(how='all', inplace=True)  # Elimina filas que son totalmente vacías
df = df.dropna(subset=[col for col in df.columns if col != 'Season'], how='all')  # Elimina filas que solo tienen 'Season'

# Suponiendo que 'Top Team Scorer' es el nombre de la columna y los datos tienen el formato 'Jugador (Goles)'
if 'Top Team Scorer' in df.columns:
    # Separar el nombre del goleador y los goles en dos columnas nuevas
    df['Scorer'] = df['Top Team Scorer'].apply(lambda x: re.findall(r"([a-zA-Z\s]+)", x)[0].strip() if pd.notna(x) else x)
    df['Goals'] = df['Top Team Scorer'].apply(lambda x: int(re.findall(r"(\d+)", x)[0]) if pd.notna(x) and re.search(r"(\d+)", x) else x)

    # Eliminar la columna original 'Top Team Scorer'
    df.drop('Top Team Scorer', axis=1, inplace=True)

print(df.head())


# Guardar los datos limpios en un nuevo archivo CSV

archivo_limpio = 'datos/limpios/23_24_limpio.csv'
df.to_csv(archivo_limpio, index=False)

    
    
# Visualizar los valores nulos
plt.figure(figsize=(15, 10))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Mapa de calor de valores nulos en el DataFrame')
plt.show()

# Mostrar la cantidad de valores nulos por columna
print("Cantidad de valores nulos por columna:")
print(df.isnull().sum())

# Opcional: Mostrar el porcentaje de valores nulos por columna
print("\nPorcentaje de valores nulos por columna:")
print(df.isnull().mean() * 100)