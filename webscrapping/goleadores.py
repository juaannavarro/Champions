from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
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
url = 'https://www.flashscore.es/futbol/europa/champions-league/clasificacion/#/KjynXKO2/top_scorers'

# Selenium navega a la página
driver.get(url)

# Espera a que la página cargue el contenido dinámico
time.sleep(5)  # Este tiempo de espera asegura que el contenido dinámico se cargue completamente

# Intenta hacer clic en "Mostrar más" hasta que no haya más datos que cargar
while True:
    try:
        # Ubica el botón "Mostrar más" y hace clic en él
        mostrar_mas_btn = driver.find_element(By.CSS_SELECTOR, 'div.topScorers__showMore')
        driver.execute_script("arguments[0].click();", mostrar_mas_btn)
        time.sleep(2)  # Espera para que los datos se carguen después de hacer clic
    except:
        break  # Si no encuentra el botón "Mostrar más", rompe el bucle

# Obtén el HTML de la página después de que se haya cargado todo el contenido dinámico
html_content = driver.page_source

# Cierra el navegador
driver.quit()

# Parsea el HTML con Beautiful Soup
soup = BeautifulSoup(html_content, 'html.parser')

# Ajusta este selector según sea necesario para capturar la tabla de goleadores
tabla_goleadores = soup.find('div', class_='topScorers__tableWrapper')

goleadores = []

if tabla_goleadores:
    filas = tabla_goleadores.find_all('div', class_='ui-table__row topScorers__row')
    for fila in filas:
        jugador_info = {}
        jugador_info['Ranking'] = fila.find('span', class_='topScorers__cell--rank').text.strip()
        jugador_info['Nombre'] = fila.find('div', class_='topScorers__participantCell--player').text.strip()
        jugador_info['Equipo'] = fila.find('a', class_='topScorers__participantCell').text.strip()
        jugador_info['Goles'] = fila.find('span', class_='topScorers__cell--goals').text.strip()
        asistencias = fila.find('span', class_='topScorers__cell--gray')
        jugador_info['Asistencias'] = asistencias.text.strip() if asistencias else '0'
        
        goleadores.append(jugador_info)

    # Guardar los datos en un archivo CSV
    with open('datos/goleadores.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['Ranking', 'Nombre', 'Equipo', 'Goles', 'Asistencias'])
        writer.writeheader()
        writer.writerows(goleadores)
    
    print("Los datos de todos los goleadores se han guardado correctamente en 'goleadores_champions_league_completo.csv'.")
else:
    print("No se encontró la tabla de goleadores.")





