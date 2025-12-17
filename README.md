FlightOnTime ✈️

Notebook de Ciencia de Datos

Predicción de retrasos de vuelos (MVP Hackathon NoCountry)

=============================

1. Imports

=============================

import pandas as pd import numpy as np from sklearn.model_selection import train_test_split from sklearn.preprocessing import OneHotEncoder from sklearn.compose import ColumnTransformer from sklearn.pipeline import Pipeline from sklearn.linear_model import LogisticRegression from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report import joblib

=============================

2. Carga del dataset

=============================

Asegurate de que el archivo flightontime_dataset.csv esté en el mismo directorio

df = pd.read_csv("flightontime_dataset.csv") print("Shape del dataset:", df.shape) df.head()

=============================

3. EDA (Análisis Exploratorio)

=============================

print("\nValores nulos por columna:") print(df.isnull().sum())

print("\nDistribución del target:") print(df['retrasado'].value_counts(normalize=True))

print("\nDescripción estadística:") print(df.describe())

=============================

4. Feature Engineering

=============================

Variables categóricas y numéricas

categorical_features = ['aerolinea', 'origen', 'destino'] numerical_features = ['hora_salida', 'dia_semana', 'distancia_km']

X = df[categorical_features + numerical_features] y = df['retrasado']

=============================

5. Preprocesamiento

=============================

preprocessor = ColumnTransformer( transformers=[ ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), ('num', 'passthrough', numerical_features) ] )

=============================

6. Modelo

=============================

model = LogisticRegression(max_iter=1000)

pipeline = Pipeline(steps=[ ('preprocessor', preprocessor), ('model', model) ])

=============================

7. Train / Test Split

=============================

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

=============================

8. Entrenamiento

=============================

pipeline.fit(X_train, y_train)

=============================

9. Evaluación

=============================

y_pred = pipeline.predict(X_test)

y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred)) print("Precision:", precision_score(y_test, y_pred)) print("Recall:", recall_score(y_test, y_pred)) print("F1-score:", f1_score(y_test, y_pred))

print("\nReporte de Clasificación:\n") print(classification_report(y_test, y_pred))

=============================

10. Guardado del modelo

=============================

joblib.dump(pipeline, "flightontime_model.joblib") print("Modelo guardado como flightontime_model.joblib")

=============================

11. Ejemplo LATAM – vuelos internacionales (país a país)

=============================

Mapeo simple aeropuerto → país (MVP)

airport_country = { 'EZE': 'Argentina', 'SCL': 'Chile', 'GIG': 'Brasil', 'GRU': 'Brasil', 'MIA': 'USA', 'JFK': 'USA', 'MAD': 'España', 'CDG': 'Francia' }

Agregamos países al dataset

df['pais_origen'] = df['origen'].map(airport_country) df['pais_destino'] = df['destino'].map(airport_country)

Filtramos solo aerolínea LATAM y vuelos internacionales

df_latam = df[ (df['aerolinea'] == 'LATAM') & (df['pais_origen'] != df['pais_destino']) ]

print("Cantidad de vuelos LATAM internacionales:", len(df_latam))

Distancia promedio entre países

distancias_por_pais = ( df_latam .groupby(['pais_origen', 'pais_destino'])['distancia_km'] .mean() .reset_index() .sort_values('distancia_km', ascending=False) )

print(" Distancia promedio (km) por ruta internacional LATAM: ") print(distancias_por_pais.head(10))

=============================

12. Predicción ejemplo LATAM (país a país)

=============================

Ejemplo: LATAM Argentina → Chile (EZE → SCL)

nuevo_vuelo_latam = pd.DataFrame([{ 'aerolinea': 'LATAM', 'origen': 'EZE', 'destino': 'SCL', 'hora_salida': 19, 'dia_semana': 4, 'distancia_km': 1140 }])

pred = pipeline.predict(nuevo_vuelo_latam)[0] prob = pipeline.predict_proba(nuevo_vuelo_latam)[0][1]

print(" Predicción LATAM internacional:") print("Ruta: Argentina → Chile") print("Estado:", "Retrasado" if pred == 1 else "Puntual") print("Probabilidad:", round(prob, 2))