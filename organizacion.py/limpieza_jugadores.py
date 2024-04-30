import pandas as pd

# Cargar los datos
archivo = 'datos/jugadores.csv'
df = pd.read_csv(archivo)
# Rellenar los valores nulos en 'Mín' con 0
df['Mín'].fillna(0, inplace=True)

# Eliminar la columna 'Año de Nacimiento'
df.drop('Año de Nacimiento', axis=1, inplace=True)
# Convertir columnas numéricas a float, manejar errores en la conversión
columnas_numericas = ['Edad', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls90.', 'Ast90', 'G+A90', 'G-TP90', 'G+A-TP90', 'xG90', 'xAG90', 'xG+xAG90', 'npxG90', 'npxG+xAG90']
for col in columnas_numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convertir a numérico, establecer 'coerce' para convertir errores a NaN

# Imputar valores nulos en columnas numéricas con la media
for col in columnas_numericas:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

# Imputar valores nulos en columnas categóricas con el valor más frecuente
for col in ['País', 'Posición']:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Eliminar las filas que solo contienen comas en su mayoría
df.dropna(how='all', inplace=True)  # Elimina filas que son totalmente vacías
df = df.dropna(subset=[col for col in df.columns if col != 'Temporada'], how='all')  # Elimina filas que solo tienen 'Temporada'

# Guardar los datos limpios en un nuevo archivo CSV
archivo_limpio = 'datos/limpios/jugadores_limpio.csv'
df.to_csv(archivo_limpio, index=False)

# Mostrar la cantidad y porcentaje de valores nulos por columna después de la limpieza
print("Cantidad de valores nulos por columna después de la limpieza:")
print(df.isnull().sum())
print("\nPorcentaje de valores nulos por columna después de la limpieza:")
print(df.isnull().mean() * 100)


