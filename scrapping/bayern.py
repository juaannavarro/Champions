from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import csv

# Configuración de Chrome para correr en modo sin cabeza
chrome_options = Options()
chrome_options.add_argument("--headless")  # Añade el argumento para correr Chrome en modo sin cabeza

# Configura Selenium con ChromeDriver y las opciones de Chrome
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL que quieres scrappear
url = 'https://fbref.com/es/equipos/054efa67/Estadisticas-de-Bayern-Munich'

# Selenium navega a la página
driver.get(url)

# Espera a que la página cargue el contenido dinámico
time.sleep(5)  # Este tiempo de espera asegura que el contenido dinámico se cargue completamente

# Obtén el HTML de la página después de que se haya cargado el contenido dinámico
html_content = driver.page_source

# Cierra el navegador
driver.quit()

# Parsea el HTML con Beautiful Soup
soup = BeautifulSoup(html_content, 'html.parser')

# Ajusta este selector según sea necesario para capturar todos los grupos
stats = soup.find_all('div', class_='table_container tabbed current is_setup', id='div_stats_standard_20')

estadisticas = []  # Lista para almacenar las estadísticas del real madrid

for stat in stats:
    rows = stat.find_all('tr')
    for row in rows:
        stats_info = {}  # Diccionario para almacenar información de las estadísticas
        th_element = row.find('th', class_='left', scope='row')
        if th_element:
            stats_info['Temporada'] = th_element.text.strip()
            data_cells = row.find_all('td')
            stats_info['Jugador'] = data_cells[0].text.strip()
            stats_info['País'] = data_cells[1].text.strip()
            stats_info['Posición'] = data_cells[2].text.strip()
            stats_info['Edad actual'] = data_cells[3].text.strip()
            stats_info['Partidos jugados'] = data_cells[4].text.strip()
            stats_info['Titular'] = data_cells[5].text.strip()
            stats_info['Minutos'] = data_cells[6].text.strip()
            stats_info['90 Jugados'] = data_cells[7].text.strip()
            stats_info['Goles'] = data_cells[8].text.strip()
            stats_info['Asistencias'] = data_cells[9].text.strip()
            stats_info['Goles + Asistencias'] = data_cells[10].text.strip()
            stats_info['Goles sin penalización'] = data_cells[11].text.strip()
            stats_info['Tiros penales ejecutados'] = data_cells[12].text.strip()
            stats_info['Tiros penales intentados'] = data_cells[13].text.strip()
            stats_info['Tarjetas amarillas'] = data_cells[14].text.strip()
            stats_info['Tarjetas rojas'] = data_cells[15].text.strip()
            stats_info['xG: Goles esperados'] = data_cells[16].text.strip()
            stats_info['npxG: Goles esperados (xG) sin contar penaltis'] = data_cells[17].text.strip()
            stats_info['xAG: Exp. Assisted Goals'] = data_cells[18].text.strip()
            stats_info['npxG + xAG'] = data_cells[19].text.strip()
            stats_info['Acarreos progresivos'] = data_cells[20].text.strip()
            stats_info['Pases progresivos'] = data_cells[21].text.strip()
            stats_info['Pases progresivos Rec'] = data_cells[22].text.strip()
            #ahora desde goles hasta npxG + xAG lo hace por 90 minutos
            stats_info['Goles por 90 minutos'] = data_cells[23].text.strip()
            stats_info['Asistencias por 90 minutos'] = data_cells[24].text.strip()
            stats_info['Goles + Asistencias por 90 minutos'] = data_cells[25].text.strip()
            stats_info['Goles sin penalización por 90 minutos'] = data_cells[26].text.strip()
            stats_info['Goles sin penalización + asistencias por 90 minutos'] = data_cells[27].text.strip()
            stats_info['xG por 90 minutos'] = data_cells[28].text.strip()
            stats_info['npxG por 90 minutos'] = data_cells[29].text.strip()
            stats_info['xAG por 90 minutos'] = data_cells[30].text.strip()
            stats_info['npxG + xAG por 90 minutos'] = data_cells[31].text.strip()
            
            estadisticas.append(stats_info)


        
        
# Guardar los datos en un archivo CSV

with open('estadisticas_bayern.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=estadisticas[0].keys())
    writer.writeheader()
    writer.writerows(estadisticas)