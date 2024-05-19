# Champions

link del repositorio: https://github.com/juaannavarro/Champions


## Descripción del Proyecto:
Este proyecto tiene como objetivo predecir el ganador de la Champions League utilizando técnicas de inteligencia artificial. Se han empleado múltiples enfoques y modelos, incluyendo regresión lineal, cadenas de Markov, clustering y redes neuronales. El énfasis principal se ha puesto en las redes neuronales debido a su capacidad de modelar complejas relaciones no lineales.

## Pasos del Proyecto
## 1. Web Scraping
El primer paso consistió en realizar web scraping en varias páginas web para obtener datos sobre jugadores, equipos y partidos. Estos datos fueron guardados en archivos CSV para su posterior procesamiento.

## 2. Limpieza de Datos
Una vez obtenidos los datos, se creó un script para su limpieza. Este paso fue crucial para asegurar que los datos estuvieran en un formato adecuado para el entrenamiento de los modelos. Los datos limpios fueron guardados en nuevos archivos CSV listos para ser utilizados.

## 3. Creación de Modelos
Se desarrollaron cuatro modelos distintos para predecir el ganador de la Champions League:

### a. Regresión Lineal
El modelo de regresión lineal se utilizó como un enfoque base para entender las relaciones lineales entre las variables.

### b. Cadenas de Markov
Las cadenas de Markov se emplearon para modelar las probabilidades de transición entre estados (por ejemplo, de una ronda a otra en el torneo).

### c. Clustering
El clustering se utilizó para agrupar equipos con características similares y analizar patrones dentro de estos grupos.

### d. Redes Neuronales
Este fue el enfoque principal del proyecto. Se utilizó una red neuronal profunda para capturar las relaciones complejas y no lineales entre las variables. El modelo fue entrenado utilizando PyTorch, una popular biblioteca de aprendizaje profundo.
