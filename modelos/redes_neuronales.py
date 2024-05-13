# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar los datos
data_23_24 = pd.read_csv('tu_ruta/23_24_limpio.csv')

# Preparar los datos
X = data_23_24.drop('Pts', axis=1)
y = data_23_24['Pts']

# Identificar columnas numéricas y categóricas
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Preprocesamiento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Crear el modelo
model = Sequential([
    Dense(128, activation='relu', input_dim=len(num_cols) + len(cat_cols)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Preparar y entrenar el modelo
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train, model__epochs=10, model__batch_size=10)

# Evaluar el modelo
score = pipeline.score(X_test, y_test)
print("Precisión del modelo:", score)
