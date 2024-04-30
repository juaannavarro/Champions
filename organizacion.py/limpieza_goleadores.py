import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
# Cargar los datos
archivo = 'datos/goleadores.csv'
df = pd.read_csv(archivo)

print(df.info())
# Mostrar la cantidad de valores nulos por columna
print("Cantidad de valores nulos por columna:")
print(df.isnull().sum())

# Opcional: Mostrar el porcentaje de valores nulos por columna
print("\nPorcentaje de valores nulos por columna:")
print(df.isnull().mean() * 100)

# Guardar los datos limpios en un nuevo archivo CSV

archivo_limpio = 'datos/limpios/goleadores_limpio.csv'
df.to_csv(archivo_limpio, index=False)

    
