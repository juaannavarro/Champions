from scrapping.goleadores import *
from scrapping.fasegrupos import *


# Asumiendo que clasificados_por_grupo y Goleadores están definidos como se discutió anteriormente

# Primero, calificar a los equipos basados en sus puntos en la fase de grupos
equipos_calificados = []
for grupo, equipos in clasificados_por_grupo.items():
    for equipo in equipos:
        equipo_info = {
            'nombre': equipo['team_name'],
            'puntos': int(equipo['points']),
            'diferencia_goles': int(equipo['goal_difference']),
            'goles_goleadores': 0  # Inicializamos esto para luego sumar los goles de los goleadores
        }
        # Sumar goles de goleadores principales si están en este equipo
        for goleador, goleador_info in Goleadores.items():
            if goleador_info['equipo'] == equipo_info['nombre']:
                equipo_info['goles_goleadores'] += goleador_info['goles']
        
        equipos_calificados.append(equipo_info)

# Ordenamos los equipos primero por puntos, luego por la suma de goles de sus goleadores, y por último por la diferencia de goles
equipos_calificados.sort(key=lambda x: (x['puntos'], x['goles_goleadores'], x['diferencia_goles']), reverse=True)

# Predecir el ganador y los siguientes tres equipos
ganador_y_siguientes = equipos_calificados[:4]

# Imprimir los resultados
print("Predicción del ganador y los siguientes tres equipos con más puntos:")
for idx, equipo in enumerate(ganador_y_siguientes, 1):
    print(f"{idx}. {equipo['nombre']} - Puntos: {equipo['puntos']}, Goles de Goleadores: {equipo['goles_goleadores']}, Diferencia de Goles: {equipo['diferencia_goles']}")

