from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import csv

# Configuración de Chrome para correr en modo sin cabeza
chrome_options = Options()
chrome_options.add_argument("--headless")

# Configura Selenium con ChromeDriver y las opciones de Chrome
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL que quieres scrappear
url = 'https://fbref.com/es/equipos/add600ae/Estadisticas-de-Dortmund'

# Selenium navega a la página
driver.get(url)

# Espera hasta que la tabla esté presente
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "matchlogs_for")))

# Obtén el HTML de la página después de que se haya cargado el contenido dinámico
html_content = driver.page_source

# Cierra el navegador
driver.quit()

# Parsea el HTML con Beautiful Soup
soup = BeautifulSoup(html_content, 'html.parser')

# Encuentra la tabla por ID
stats_table = soup.find('table', id="matchlogs_for")
rows = stats_table.find_all('tr')[1:]  # Excluye el encabezado

estadisticas = []  # Lista para almacenar las estadísticas

for row in rows:
    cells = row.find_all('td')
    if cells:
        stats_info = {
            'Fecha': row.find('th', {'scope': 'row'}).text.strip(),
            'Hora': cells[0].text.strip(),
            'Competición': cells[1].text.strip(),
            'Ronda': cells[2].text.strip(),
            'Día': cells[3].text.strip(),
            'Sede': cells[4].text.strip(),
            'Resultado': cells[5].text.strip(),
            'Goles a favor': cells[6].text.strip(),
            'Goles en contra': cells[7].text.strip(),
            'Rival': cells[8].text.strip(),
            'xG': cells[9].text.strip() if len(cells) > 9 else '',
            'xGA': cells[10].text.strip() if len(cells) > 10 else '',
            'Posesión': cells[11].text.strip() if len(cells) > 11 else '',
            'Asistencia': cells[12].text.strip() if len(cells) > 12 else '',
            'Capitán': cells[13].text.strip() if len(cells) > 13 else '',
            'Formación': cells[14].text.strip() if len(cells) > 14 else '',
            'Árbitro': cells[15].text.strip() if len(cells) > 15 else ''
        }
        estadisticas.append(stats_info)

# Guardar los datos en un archivo CSV
if estadisticas:
    with open('datos/Resultados_dortmund.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=estadisticas[0].keys())
        writer.writeheader()
        for data in estadisticas:
            writer.writerow(data)

print("Datos recopilados y guardados.")
