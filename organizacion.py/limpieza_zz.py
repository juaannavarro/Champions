import pandas as pd

def clean_csv(input_path, output_path):
    # Cargar los datos
    df = pd.read_csv(input_path)
    

    
    # Eliminar duplicados
    df.drop_duplicates(inplace=True)

    df.dropna(inplace=True)
    
    # Guardar a un nuevo archivo CSV
    df.to_csv(output_path, index=False)
    print(f"Archivo limpiado guardado como: {output_path}")

# Rutas de los archivos de entrada y salida
input_csv_path = 'datos/zz.csv'  # Asegúrate de ajustar esta ruta al archivo generado
output_csv_path = 'datos/zz_cleaned.csv'

# Llamar a la función de limpieza
clean_csv(input_csv_path, output_csv_path)
