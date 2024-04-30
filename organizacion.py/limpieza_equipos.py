import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Cargar los datos
archivo = 'datos/equipos.csv'
df = pd.read_csv(archivo)

# Eliminar las columnas Time y Match Report

df.drop(['Time', 'Match Report'], axis=1, inplace=True)

# Eliminar filas que son completamente vacías o solo contienen 'Season'

df.dropna(how='all', inplace=True)  # Elimina filas que son totalmente vacías
df = df.dropna(subset=[col for col in df.columns if col != 'Temporada'], how='all')  # Elimina filas que solo tienen 'Season'






# Guardar los datos limpios en un nuevo archivo CSV

archivo_limpio = 'datos/limpios/equipos_limpio.csv'
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


#Borramos las columnas con valores nulos, al ejecutar el codigo anterior vemos que son Attendance y Notes

df.drop(['Attendance', 'Notes'], axis=1, inplace=True)

#ejecutamos de nuevo


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
