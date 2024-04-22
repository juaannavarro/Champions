from scrapping.goleadores import *
from scrapping.fasegrupos import *
import random

# Función para simular los resultados de los partidos basados en la fuerza del equipo y goleadores
def simular_partido(equipo1, equipo2, goleadores, clasificados):
    # Calculamos la fuerza de cada equipo basada en su rendimiento en la fase de grupos y sus goleadores
    fuerza_equipo1 = calcular_fuerza(equipo1, goleadores, clasificados)
    fuerza_equipo2 = calcular_fuerza(equipo2, goleadores, clasificados)
    
    # Simulamos el resultado basándonos en la fuerza calculada
    resultado = random.choices([equipo1, equipo2, 'empate'], weights=[fuerza_equipo1, fuerza_equipo2, 1])[0]
    
    return resultado

# Función para calcular la fuerza de un equipo
def calcular_fuerza(equipo, goleadores, clasificados):
    fuerza = 0
    # Añadir puntos por clasificación y goles de goleadores
    for grupo, equipos in clasificados.items():
        for e in equipos:
            if e['team_name'] == equipo:
                fuerza += int(e['points'])  # Puntos de fase de grupos
                break
    
    for goleador, datos in goleadores.items():
        if datos['equipo'] == equipo:
            fuerza += datos['goles']  # Añadir fuerza por goles de goleadores
    
    return fuerza
# Definición inicial de los enfrentamientos de octavos de final
octavos = [
    ("Lazio", "Bayern"),
    ("Inter de Milan", "Atlético de Madrid"),
    ("Napoli", "Barcelona"),
    ("Oporto", "Arsenal"),
    ("PSV", "Dortmund"),
    ("PSG", "Real Sociedad"),
    ("Copenhague", "Manchester City"),
    ("RB Leipzig", "Real Madrid")
]

# Simular los octavos de final y guardar los ganadores
print("Octavos de Final:")

ganadores_octavos = []
for equipo1, equipo2 in octavos:
    ganador = simular_partido(equipo1, equipo2, Goleadores, clasificados_por_grupo)
    print(f"Resultado de {equipo1} vs {equipo2}: {ganador}")
    if ganador != 'empate':  # Asumimos que manejamos los empates eligiendo un ganador al azar
        ganadores_octavos.append(ganador)
    else:
        ganadores_octavos.append(random.choice([equipo1, equipo2]))

# Función para sortear y simular rondas
def sortear_y_simular(ganadores_previos):
    random.shuffle(ganadores_previos)
    parejas = [ganadores_previos[i:i + 2] for i in range(0, len(ganadores_previos), 2)]
    ganadores = []
    for equipo1, equipo2 in parejas:
        ganador = simular_partido(equipo1, equipo2, Goleadores, clasificados_por_grupo)
        print(f"Resultado de {equipo1} vs {equipo2}: {ganador}")
        if ganador != 'empate':  # Manejo de empates
            ganadores.append(ganador)
        else:
            ganadores.append(random.choice([equipo1, equipo2]))
    return ganadores

# Simular cuartos de final
print("Cuartos de Final:")

ganadores_cuartos = sortear_y_simular(ganadores_octavos)

# Simular semifinales
print("Semifinal:")

ganadores_semis = sortear_y_simular(ganadores_cuartos)

# Simular la final
print("Final:")

finalistas = [ganadores_semis[i] for i in range(2)]
ganador_final = simular_partido(finalistas[0], finalistas[1], Goleadores, clasificados_por_grupo)
print(f"El ganador del torneo es: {ganador_final}")

