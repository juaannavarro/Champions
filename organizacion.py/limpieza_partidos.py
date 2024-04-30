import pandas as pd
import re

def clean_score(score):
    if pd.isna(score):
        return None
    match = re.search(r'(\d+)[–-](\d+)', str(score))
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None

# Cargar los datos
archivo = 'datos/partidos23_24.csv'
df = pd.read_csv(archivo)

# Eliminar las filas que solo contienen comas en su mayoría
df.dropna(how='all', inplace=True)  # Elimina filas que son totalmente vacías
df = df.dropna(subset=[col for col in df.columns if col != '2023-2024'], how='all')  # Elimina filas que solo tienen 'Season'

# Aplicar la función clean_score a la columna 'Score'
df['Score'] = df['Score'].apply(clean_score)

# Eliminar columnas 'Attendance' y 'Referee'
df.drop(['Attendance', 'Referee'], axis=1, inplace=True)

# Eliminar filas donde 'Score' es None después de la limpieza
df.dropna(subset=['Score'], inplace=True)

# Guardar los datos limpios en un nuevo archivo CSV
archivo_limpio = 'datos/limpios/partidos23_24_limpio.csv'
df.to_csv(archivo_limpio, index=False)

# Mostrar la cantidad de valores nulos por columna
print("Cantidad de valores nulos por columna:")
print(df.isnull().sum())

# Opcional: Mostrar el porcentaje de valores nulos por columna
print("\nPorcentaje de valores nulos por columna:")
print(df.isnull().mean() * 100)
