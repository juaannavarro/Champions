from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time

# Lista de temporadas
temporadas = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']

# Configuración del navegador
options = Options()
options.add_argument('--headless')  # Para ejecutar el navegador en segundo plano (sin interfaz gráfica)
driver = webdriver.Chrome(options=options)

# Lista para almacenar todas las filas de datos
datos_grandes = []

# Columnas a extraer, actualizado según tu requerimiento
encabezado = ['Temporada', '#', 'Jugador', 'País', 'Posición', 'Equipo', 'Edad', 'Año de Nacimiento',
              'PJ', 'Titular', 'Mín', '90 s', 'Gls.', 'Ass', 'G+A', 'G-TP', 'TP', 'TPint',
              'TA', 'TR', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls90.', 'Ast90',
              'G+A90', 'G-TP90', 'G+A-TP90', 'xG90', 'xAG90', 'xG+xAG90', 'npxG90', 'npxG+xAG90', 'Partidos']

# Iterar sobre las temporadas
for temporada in temporadas:
    # Construir la URL para la temporada actual
    url = f'https://fbref.com/es/comps/8/{temporada}/stats/Estadisticas-{temporada}-Champions-League'

    # Obtener la página web con Selenium
    driver.get(url)

    # Esperar a que la página cargue completamente
    time.sleep(5)  # Asegura que el contenido dinámico se haya cargado

    # Obtener el contenido HTML
    html_content = driver.page_source

    # Utilizar BeautifulSoup para analizar el HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extraer los datos de la tabla correcta utilizando el ID específico y las clases mencionadas
    stats_table = soup.find('table', class_="min_width sortable stats_table shade_zero now_sortable sticky_table eq2 re2 le2", id="stats_standard")
    if stats_table:
        for row in stats_table.find_all('tr')[1:]:  # Ignorar el encabezado de la tabla
            cells = row.find_all(['th', 'td'])
            if len(cells) > 1:  # Asegura que la fila no sea solo de encabezado
                row_data = [temporada] + [cell.get_text(strip=True) for cell in cells]
                if len(row_data) == len(encabezado):  # Solo agregar filas completas
                    datos_grandes.append(row_data)

# Cerrar el navegador
driver.quit()

# Crear un DataFrame con los datos recopilados
try:
    df = pd.DataFrame(datos_grandes, columns=encabezado)
    # Guardar el DataFrame en un archivo CSV
    archivo_csv = 'datos/jugadores.csv'
    df.to_csv(archivo_csv, index=False, encoding='utf-8')
    print(f"Datos guardados en {archivo_csv}")
except Exception as e:
    print(f"Error al crear DataFrame: {e}")



