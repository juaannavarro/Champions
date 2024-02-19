from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

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
clasificados_por_grupo = {}  # Diccionario para almacenar los dos primeros de cada grupo

for grupo_index, grupo in enumerate(grupos):
    # Inicializa una lista vacía para los dos primeros equipos de este grupo
    clasificados_por_grupo[letras_del_grupo[grupo_index]] = []
    
    rows = grupo.find_all('div', class_='ui-table__row')[:2]  # Limita a los dos primeros equipos

    for row in rows:
        equipo_info = {}  # Diccionario para almacenar información del equipo
        
        equipo_info['rank'] = row.find('div', class_='table__cell--rank').text.strip()
        equipo_info['team_name'] = row.find('a', class_='tableCellParticipant__name').text.strip()
        
        # Extracción de otros datos como partidos jugados, victorias, empates, derrotas, etc.
        data_cells = row.find_all('span', class_='table__cell--value')
        equipo_info['matches_played'] = data_cells[0].text.strip()
        equipo_info['wins'] = data_cells[1].text.strip()
        equipo_info['draws'] = data_cells[2].text.strip()
        equipo_info['losses'] = data_cells[3].text.strip()
        equipo_info['goals'] = data_cells[4].text.strip()  # "goles a favor:goles en contra"
        equipo_info['goal_difference'] = data_cells[5].text.strip()
        equipo_info['points'] = data_cells[6].text.strip()
        
        # Añade el equipo actual a la lista del grupo correspondiente
        clasificados_por_grupo[letras_del_grupo[grupo_index]].append(equipo_info)
