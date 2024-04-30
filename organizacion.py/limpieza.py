import pandas as pd
def standardize_headers(df):
    header_mapping = {
        'Rk': 'Rank',
        'Squad': 'Team',
        'MP': 'Matches_Played',
        'PJ': 'Matches_Played',
        'W': 'Wins',
        'D': 'Draws',
        'L': 'Losses',
        'GF': 'Goals_For',
        'GA': 'Goals_Against',
        'GD': 'Goal_Difference',
        'Pts': 'Points',
        'xG': 'Expected_Goals',
        'xGA': 'Expected_Goals_Against',
        'xGD': 'Expected_Goal_Difference',
        'xGD/90': 'Expected_Goal_Difference_Per_90',
        'Last 5': 'Last_5_Matches',
        'Attendance': 'Attendance',
        'Top Team Scorer': 'Top_Scorer',
        'Goalkeeper': 'Goalkeeper',
        'Notes': 'Notes',
        'Temporada': 'Season',
        'Round': 'Round',
        'Day': 'Day',
        'Date': 'Date',
        'Time': 'Time',
        'Home': 'Home_Team',
        'Score': 'Score',
        'Away': 'Away_Team',
        'Venue': 'Venue',
        'Referee': 'Referee',
        'Match Report': 'Match_Report',
        'Nombre': 'Player_Name',
        'Equipo': 'Team',
        'Goles': 'Goals',
        'Asistencias': 'Assists',
        'Jugador': 'Player',
        'País': 'Country',
        'Posición': 'Position',
        'Edad': 'Age',
        'Año de Nacimiento': 'Year_of_Birth',
        'Mín': 'Minutes_Played',
        '90 s': 'Full_Matches_Played',
        'Goles.': 'Goals',
        'G+A': 'Goals_and_Assists',
        'G-TP': 'Goals_from_Open_Play',
        'TP': 'Penalties',
        'TPint': 'Penalties_Intercepted',
        'TA': 'Yellow_Cards',
        'TR': 'Red_Cards',
        'xG90': 'Expected_Goals_Per_90',
        'xAG90': 'Expected_Assists_Per_90',
        'xG+xAG90': 'Expected_Goals_and_Assists_Per_90',
        'npxG90': 'Non_Penalty_Expected_Goals_Per_90',
        'npxG+xAG90': 'Expected_Non_Penalty_Goals_and_Assists_Per_90',
        'Partidos': 'Matches',
        'PrgC': 'Progressive_Carries',
        'PrgP': 'Progressive_Passes',
        'PrgR': 'Progressive_Runs'
    }
    df.rename(columns=header_mapping, inplace=True)
    return df
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: El archivo no fue encontrado en {file_path}")
        return None

def clean_data(df, critical_columns):
    if df is None:
        return None
    print(df.isna().sum())
    df.dropna(subset=critical_columns, inplace=True)
    if 'Squad' in df.columns:
        df['Squad'] = df['Squad'].apply(lambda x: x.split(' ')[-1])
    df = standardize_headers(df)
    return df


def save_clean_data(df, file_name):
    df.to_csv(file_name, index=False)

# Cargar los datos
data_files = {
    '23_24': 'datos/23_24.csv',
    'equipos': 'datos/equipos.csv',
    'goleadores': 'datos/goleadores.csv',
    'jugadores': 'datos/jugadores.csv'
}

cleaned_data = {}

# Limpieza de datos
for name, path in data_files.items():
    df = load_data(path)
    # Asumiendo que 'Pts' y 'Squad' son columnas críticas para '23_24.csv'
    critical_columns = ['Pts', 'Squad'] if name == '23_24' else None
    cleaned_df = clean_data(df, critical_columns)
    save_clean_data(cleaned_df, f'datos/datos_limpios/cleaned_{name}.csv')
    cleaned_data[name] = cleaned_df


