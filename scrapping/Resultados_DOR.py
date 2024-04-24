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
url = 'https://fbref.com/es/equipos/add600ae/Estadisticas-de-Dortmund'

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
stats = soup.find_all('div', class_='table_container tabbed current is_setup', id="div_matchlogs_for")

estadisticas = []  # Lista para almacenar las estadísticas del real madrid

for stat in stats:
    rows = stat.find_all('tr')
    for row in rows:
        stats_info = {}  # Diccionario para almacenar información de las estadísticas
        th_element = row.find('th', class_='left', scope='row')
        if th_element:
            stats_info['Temporada'] = th_element.text.strip()
            data_cells = row.find_all('td')
            stats_info['Fecha'] = data_cells[0].text.strip()
            stats_info['Ronda'] = data_cells[1].text.strip()
            stats_info['Día'] = data_cells[2].text.strip()
            stats_info['Sedes'] = data_cells[3].text.strip()
            stats_info['Resultado'] = data_cells[4].text.strip()
            stats_info['Goles a favor'] = data_cells[5].text.strip()
            stats_info['Goles en contra'] = data_cells[6].text.strip()
            stats_info['Adversario'] = data_cells[7].text.strip()
            stats_info['xG'] = data_cells[8].text.strip()
            stats_info['xGA'] = data_cells[9].text.strip()
            stats_info['Posesión'] = data_cells[10].text.strip()
            stats_info['Asistencia'] = data_cells[11].text.strip()
            stats_info['Capitán'] = data_cells[12].text.strip()
            stats_info['Formación'] = data_cells[13].text.strip()
            stats_info['Árbitro'] = data_cells[14].text.strip()
            stats_info['Informe'] = data_cells[15].text.strip()
            stats_info['Notas'] = data_cells[16].text.strip()
            estadisticas.append(stats_info)


        
        
# Guardar los datos en un archivo CSV

with open('resultados_dormund.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=estadisticas[0].keys())
    writer.writeheader()
    writer.writerows(estadisticas)