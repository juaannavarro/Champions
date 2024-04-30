import requests
from bs4 import BeautifulSoup
import csv

# URL de la página de donde extraeremos los datos
url = 'https://fbref.com/en/comps/8/Champions-League-Stats'

# Realizamos la petición HTTP a la página
response = requests.get(url)
response.raise_for_status()  # Asegurarse de que la respuesta es exitosa

# Parseamos el contenido HTML con BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Buscamos la tabla por el ID específico
table = soup.find('table', id='results2023-202480_overall')

# Preparamos una lista para guardar los datos extraídos
data_rows = []

# Verificamos que la tabla fue encontrada
if table:
    # Extraemos las filas de la tabla
    rows = table.find_all('tr')
    for row in rows:
        # Obtenemos las celdas de cada fila (th o td)
        cells = row.find_all(['th', 'td'])
        # Extraemos el texto de cada celda y lo agregamos a la lista de la fila
        row_data = [cell.get_text(strip=True) for cell in cells]
        data_rows.append(row_data)

# Nombre del archivo CSV donde se guardarán los datos
output_csv = 'datos/23_24.csv'

# Guardamos los datos en un archivo CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_rows)

print(f"Los datos han sido guardados en {output_csv}")
