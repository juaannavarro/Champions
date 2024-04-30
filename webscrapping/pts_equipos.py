import csv
import requests
from bs4 import BeautifulSoup

def fetch_data_for_year(year):
    """Fetch and parse table data for a given UEFA Champions League season."""
    url = f'https://fbref.com/en/comps/8/{year}/{year}-Champions-League-Stats'
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('span', attrs={'data-label': 'League Table'}).find_next('table', id=lambda x: x and 'overall' in x)

    return table.find_all('tr')

def process_rows(rows, year, all_data):
    """Process table rows and append them to the list with proper formatting."""
    for row in rows:
        cells = [cell for cell in row.find_all(['th', 'td']) if 'xg' not in cell.get('data-stat')]
        data_row = [cell.get_text(strip=True) for cell in cells]
        if data_row:
            if not all_data:
                data_row.insert(0, 'Season')
            elif data_row != all_data[0][1:]:
                data_row.insert(0, year)
            all_data.append(data_row)

def save_data_to_csv(data, filepath):
    """Save the collected data into a CSV file."""
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
    print(f"El archivo CSV '{filepath}' ha sido creado exitosamente.")

def main():
    years = ["2022-2023", "2021-2022", "2020-2021", "2019-2020", "2018-2019", "2017-2018",
             "2016-2017", "2015-2016", "2014-2015", "2013-2014", "2012-2013", "2011-2012",
             "2010-2011", "2009-2010", "2008-2009", "2007-2008", "2006-2007", "2005-2006",
             "2004-2005", "2003-2004"]

    all_data = []

    for year in years:
        try:
            rows = fetch_data_for_year(year)
            process_rows(rows, year, all_data)
        except Exception as e:
            print(f"Error al procesar la temporada {year}: {e}")

    save_data_to_csv(all_data, 'datos/pts_equipos.csv')

if __name__ == '__main__':
    main()
