import pandas as pd
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



# Función para determinar el resultado del partido basado en el marcador
def determine_outcome(score):
    home_goals, away_goals = map(int, score.split('-'))
    if home_goals > away_goals:
        return 'W', 'L'  # Local gana, Visitante pierde
    elif home_goals < away_goals:
        return 'L', 'W'  # Local pierde, Visitante gana
    else:
        return 'D', 'D'  # Empate

# Cargar datos de partidos
partidos_data = pd.read_csv("datos/limpios/partidos23_24_limpio.csv")
partidos_data['Home Result'], partidos_data['Away Result'] = zip(*partidos_data['Score'].apply(determine_outcome))

# Inicializar diccionario para contar transiciones para equipos locales y visitantes
transitions = defaultdict(lambda: defaultdict(int))
team_states = {}  # Almacenar el último estado de cada equipo

# Actualizar los conteos de transiciones
for _, match in partidos_data.iterrows():
    for team_type in ['Home', 'Away']:
        current_result = match[team_type + ' Result']
        team = match[team_type]
        prev_result = team_states.get(team, None)
        if prev_result is not None:
            transitions[prev_result][current_result] += 1
        team_states[team] = current_result

# Calcular probabilidades de transición
transition_probabilities = {state: {result: count / sum(counts.values()) 
                                     for result, count in counts.items()} 
                             for state, counts in transitions.items()}

print(transition_probabilities)  # Imprimir las probabilidades de transición


# Simulación de la ejecución del código proporcionado, reutilizando los datos ya cargados y procesados previamente

# Re-Calculating the transition probabilities from the previously processed data
transitions_simulated = defaultdict(lambda: defaultdict(int))
team_states_simulated = {}

# Updating the transition counts as per the provided script
for _, match in partidos_data.iterrows():
    for team_type in ['Home', 'Away']:
        current_result = match[team_type + ' Result']
        team = match[team_type]
        prev_result = team_states_simulated.get(team, None)
        if prev_result is not None:
            transitions_simulated[prev_result][current_result] += 1
        team_states_simulated[team] = current_result

# Calculating transition probabilities as per the provided script
transition_probabilities_simulated = {state: {result: count / sum(counts.values()) 
                                              for result, count in counts.items()} 
                                      for state, counts in transitions_simulated.items()}

# Using the same graph creation logic as before to visualize the transition probabilities
G_simulated = nx.DiGraph()
positions_simulated = {'W': (1, 1), 'D': (2, 0), 'L': (3, 1)}

# Add nodes and edges with probabilities as weights
for state, pos in positions_simulated.items():
    G_simulated.add_node(state, pos=pos)

for start_state, transitions in transition_probabilities_simulated.items():
    for end_state, probability in transitions.items():
        G_simulated.add_edge(start_state, end_state, weight=round(probability, 2))

# Draw the graph
plt.figure(figsize=(8, 5))
nx.draw(G_simulated, pos=positions_simulated, with_labels=True, node_color='skyblue', node_size=4000, 
        edge_color='gray', width=2, font_size=15, font_color='black')
edge_labels_simulated = {(start, end): G_simulated.edges[start, end]['weight'] for start, end in G_simulated.edges}
nx.draw_networkx_edge_labels(G_simulated, pos=positions_simulated, edge_labels=edge_labels_simulated, font_color='red', font_size=12)

plt.title('Cadena de Markov Simulada: Probabilidades de Transición de Resultados de Partidos')
plt.show()


# Define the states in the order we want them to appear in the matrix
states = ['W', 'D', 'L']

# Initialize the transition matrix with zeros
transition_matrix = np.zeros((len(states), len(states)))

# Fill the matrix with the calculated probabilities
for i, origin_state in enumerate(states):
    for j, destination_state in enumerate(states):
        transition_matrix[i, j] = transition_probabilities.get(origin_state, {}).get(destination_state, 0)

# Create a DataFrame for better visualization and indexing
transition_matrix_df = pd.DataFrame(transition_matrix, index=states, columns=states)

print("\nMatriz de Transición de Resultados de Partidos:")
print(transition_matrix_df)  # Display the transition matrix