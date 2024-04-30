import csv
import requests
from bs4 import BeautifulSoup

def fetch_and_parse_data(year):
    """Fetch and parse Champions League schedule data for a given year."""
    url = f'https://fbref.com/en/comps/8/schedule/-Champions-League-Scores-and-Fixtures'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('span', attrs={'data-label': 'Scores & Fixtures'}).find_next('table', id=lambda x: x and '_8_3' in x)
    return table.find_all('tr')

def extract_data_from_rows(rows, year):
    """Extract necessary data from table rows."""
    data = []
    for row in rows:
        cells = [cell for cell in row.find_all(['th', 'td']) if 'xg' not in cell.get('data-stat')]
        row_data = [cell.get_text(strip=True) for cell in cells]
        if row_data:
            # Insert season info only if it's not the header row or a redundant header row
            if not data or (data and row_data != data[0][1:]):
                row_data.insert(0, year)
            data.append(row_data[:17])  # Assuming you need the first 17 elements
    return data

def save_data_to_csv(data, filepath):
    """Save the extracted data to a CSV file."""
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
    print(f"The CSV file '{filepath}' has been created successfully.")

def main():
    years = ["2023-2024"]  # Example: Extend this list for more years
    all_data = []

    for year in years:
        rows = fetch_and_parse_data(year)
        year_data = extract_data_from_rows(rows, year)
        all_data.extend(year_data)

    save_data_to_csv(all_data, 'datos/partidos23_24.csv')

if __name__ == '__main__':
    main()
