import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
# Cargar los datos
archivo = 'datos/equipos.csv'
df = pd.read_csv(archivo)
# Función para determinar si una fila es un encabezado repetido
def es_encabezado_repetido(row):
    # Concatenar todos los elementos de la fila en una cadena
    fila_concatenada = ','.join(map(str, row))
    # Verificar si la fila sigue el patrón de encabezado con cualquier año
    match = re.match(r'\d{4}-\d{4},Round,Day,Date,Home,Score,Away,Attendance,Venue,Referee,Notes', fila_concatenada)
    return bool(match)  # Asegura devolver True si hay coincidencia, False si no

# Aplicar la función a cada fila y filtrar las filas que no son encabezados repetidos
df = df[~df.apply(es_encabezado_repetido, axis=1)]
# Eliminar las columnas Time y Match Report

df.drop(['Time', 'Match Report'], axis=1, inplace=True)

# Eliminar filas que son completamente vacías o solo contienen 'Season'

df.dropna(how='all', inplace=True)  # Elimina filas que son totalmente vacías
df = df.dropna(subset=[col for col in df.columns if col != 'Temporada'], how='all')  # Elimina filas que solo tienen 'Season'






# Guardar los datos limpios en un nuevo archivo CSV

archivo_limpio = 'datos/limpios/equipos_limpio.csv'
df.to_csv(archivo_limpio, index=False)

    


# Mostrar la cantidad de valores nulos por columna
print("Cantidad de valores nulos por columna:")
print(df.isnull().sum())

# Opcional: Mostrar el porcentaje de valores nulos por columna
print("\nPorcentaje de valores nulos por columna:")
print(df.isnull().mean() * 100) 


#Borramos las columnas con valores nulos, al ejecutar el codigo anterior vemos que son Attendance y Notes

df.drop(['Attendance', 'Notes'], axis=1, inplace=True)

#ejecutamos de nuevo




# Mostrar la cantidad de valores nulos por columna
print("Cantidad de valores nulos por columna:")
print(df.isnull().sum())

# Opcional: Mostrar el porcentaje de valores nulos por columna
print("\nPorcentaje de valores nulos por columna:")
print(df.isnull().mean() * 100) 
