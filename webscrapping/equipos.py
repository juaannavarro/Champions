from bs4 import BeautifulSoup
import requests
import csv

# Definimos los años de interés para la recopilación de datos
periodos = [
    '2017-2018', '2018-2019', '2019-2020', '2020-2021', 
    '2021-2022', '2022-2023'
]

# Lista para acumular información de todas las temporadas
informacion_completa = []

def extraer_datos(temporada):
    try:
        url = f'https://fbref.com/en/comps/8/{temporada}/schedule/{temporada}-Champions-League-Scores-and-Fixtures'
        respuesta = requests.get(url)
        respuesta.raise_for_status()  # Verifica errores en la solicitud

        analisis = BeautifulSoup(respuesta.text, 'html.parser')
        tabla_resultados = analisis.find('span', attrs={'data-label': 'Scores & Fixtures'}).find_next('table', id=lambda x: x and '_8_3' in x)

        # Procesamos las filas de la tabla
        filas = tabla_resultados.find_all('tr')
        for fila in filas:
            celdas = fila.find_all(['th', 'td'])
            # Filtramos celdas para excluir aquellas que contienen 'xg'
            celdas_filtradas = [celda for celda in celdas if 'xg' not in celda.get('data-stat')]
            # Recolectamos datos de cada celda
            datos_fila = [celda.get_text(strip=True) for celda in celdas_filtradas]
            datos_fila.insert(0, temporada)  # Añadimos temporada al inicio de cada fila
            informacion_completa.append(datos_fila)

    except requests.exceptions.RequestException as error:
        print(f"Error al recopilar datos de la temporada {temporada}: {error}")

# Procesar cada temporada
for periodo in periodos:
    extraer_datos(periodo)

# Cambiamos el encabezado de la primera columna
informacion_completa[0][0] = 'Temporada'
# Escribir los datos recopilados en un archivo CSV
archivo_csv = 'datos/equipos.csv'
with open(archivo_csv, 'w', newline='', encoding='utf-8') as archivo:
    escritor = csv.writer(archivo)
    for informacion in informacion_completa:
        escritor.writerow(informacion)

# Mensaje de confirmación de creación del archivo
print(f"Archivo CSV '{archivo_csv}' creado exitosamente.")



