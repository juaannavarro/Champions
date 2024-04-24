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
url = 'https://www.flashscore.es/futbol/europa/champions-league/clasificacion/#/OvQi9HWe/table/overall'

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
grupos = soup.find_all('div', class_='ui-table__body')

letras_del_grupo = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
clasificados_por_grupo = []  # Lista para almacenar todos los equipos clasificados con su grupo correspondiente

for grupo_index, grupo in enumerate(grupos):
    rows = grupo.find_all('div', class_='ui-table__row')[:2]  # Limita a los dos primeros equipos
    for row in rows:
        equipo_info = {}  # Diccionario para almacenar información del equipo
        equipo_info['Grupo'] = letras_del_grupo[grupo_index]
        equipo_info['Rank'] = row.find('div', class_='table__cell--rank').text.strip()
        equipo_info['Team Name'] = row.find('a', class_='tableCellParticipant__name').text.strip()
        data_cells = row.find_all('span', class_='table__cell--value')
        equipo_info['Matches Played'] = data_cells[0].text.strip()
        equipo_info['Wins'] = data_cells[1].text.strip()
        equipo_info['Draws'] = data_cells[2].text.strip()
        equipo_info['Losses'] = data_cells[3].text.strip()
        equipo_info['Goals'] = data_cells[4].text.strip()  # "goles a favor:goles en contra"
        equipo_info['Goal Difference'] = data_cells[5].text.strip()
        equipo_info['Points'] = data_cells[6].text.strip()
        
        clasificados_por_grupo.append(equipo_info)

# Guardar los datos en un archivo CSV
with open('clasificacion_fase_grupos_champions_league.csv', 'w', newline='', encoding='utf-8') as file:
    fieldnames = ['Grupo', 'Ranking', 'Equipo', 'Partidos Jugados', 'Victorias', 'Derrotas', 'Empate', 'Goles', 'Diferencia de Goles', 'Puntos']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(clasificados_por_grupo)

print("La clasificación de la fase de grupos se ha guardado correctamente en 'clasificacion_fase_grupos_champions_league.csv'.")

