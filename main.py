from fastapi import FastAPI
import pandas as pd 
from datetime import datetime
import locale
import nltk 
from nltk.tokenize import word_tokenize
import ast
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
from io import StringIO

url = 'https://raw.githubusercontent.com/FFenix43/Proyecto-Individual-Henry/main/movies_dataset_modificado.csv'
response = requests.get(url)
data = StringIO(response.text)
dataset = pd.read_csv(data)
url = 'https://www.dropbox.com/s/f1oxmgytzlyknkr/credits.csv?dl=1'  # Cambiar dl=0 a dl=1
response = requests.get(url)
data2 = StringIO(response.text)
dataset2 = pd.read_csv(data2)

dataset = dataset.dropna(subset=["overview"])
dataset["overview"].fillna("", inplace=True)


def filter_non_numeric_ids(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

dataset = dataset[dataset['id'].apply(filter_non_numeric_ids)]
dataset['id'] = dataset['id'].astype('int64')

nltk.download('punkt')

app = FastAPI()

def cantidad_filmaciones_mes(mes):
    # Convertir el mes en español a número de mes
    locale.setlocale(locale.LC_TIME, "es_ES")
    mes_numero = datetime.strptime(mes, "%B").month

    dataset_cleaned = dataset.dropna(subset=['release_date'])

    # Obtener el mes de la columna release_date
    dataset_cleaned['release_month'] = pd.to_datetime(dataset_cleaned['release_date']).dt.month

    # Filtrar el dataset para obtener las películas estrenadas en el mes consultado
    peliculas_mes = dataset_cleaned[dataset_cleaned['release_month'] == mes_numero]

    # Obtener la cantidad de películas encontradas
    cantidad = len(peliculas_mes)

    return cantidad


@app.get("/cantidad_filmaciones/{mes}")
def obtener_cantidad_filmaciones_mes(mes: str):
    cantidad = cantidad_filmaciones_mes(mes)
    respuesta = f"{cantidad} es la cantidad de peliculas que fueron estrenadas en el mes de {mes.capitalize()}"
    return respuesta



def cantidad_filmaciones_dia_semana(dia_semana):
    dataset_cleaned = dataset.dropna(subset=['release_date'])
    dataset_cleaned['release_date'] = pd.to_datetime(dataset_cleaned['release_date'])

    # Obtener el nombre del día de la semana en español
    dias_semana_espanol = {
        0: 'lunes',
        1: 'martes',
        2: 'miercoles',
        3: 'jueves',
        4: 'viernes',
        5: 'sabado',
        6: 'domingo'
    }

    # Filtrar el dataset para obtener las películas estrenadas en el día de la semana consultado
    peliculas_dia_semana = dataset_cleaned[dataset_cleaned['release_date'].dt.dayofweek == dia_semana]

    # Obtener la cantidad de películas encontradas
    cantidad = len(peliculas_dia_semana)

    return cantidad, dias_semana_espanol[dia_semana]


@app.get("/cantidad_filmaciones_por_dia/{dia_semana}")
def obtener_cantidad_filmaciones_dia_semana(dia_semana: str):
    dias_semana_espanol = {
        'lunes': 0,
        'martes': 1,
        'miercoles': 2,
        'jueves': 3,
        'viernes': 4,
        'sabado': 5,
        'domingo': 6
    }

    dia_semana_lower = dia_semana.lower()
    if dia_semana_lower not in dias_semana_espanol:
        return {"error": "Día de la semana inválido"}

    dia_semana_num = dias_semana_espanol[dia_semana_lower]
    cantidad, nombre_dia = cantidad_filmaciones_dia_semana(dia_semana_num)
    mensaje = f"{cantidad} cantidad de películas fueron estrenadas en el día {nombre_dia}"

    return {"mensaje": mensaje}

@app.get("/score_titulo/{titulo}")
def obtener_info_pelicula(titulo: str):
    # Filtrar el dataset por título de la película
    pelicula = dataset[dataset['title'] == titulo]

    # Verificar si la película existe en el dataset
    if pelicula.empty:
        return {"mensaje": "La película no fue encontrada"}

    # Obtener el título, año de estreno y puntaje de la película
    titulo_pelicula = pelicula['title'].iloc[0]
    anio_estreno = int(pelicula['release_year'].iloc[0])
    score = pelicula['popularity'].iloc[0]

    mensaje = f"La película {titulo_pelicula} fue estrenada en el año {anio_estreno} con un score de {score}."
    return {"mensaje": mensaje}

@app.get("/votos_titulo/{titulo}")
def obtener_info_votos(titulo: str):
    # Filtrar el dataset por título de la filmación
    filmacion = dataset[dataset['title'] == titulo]

    # Verificar si la filmación existe en el dataset
    if filmacion.empty:
        return {"mensaje": "La filmación no fue encontrada"}

    # Obtener la cantidad de votos y el valor promedio de las votaciones
    votos = filmacion['vote_count'].iloc[0]
    promedio_votos = filmacion['vote_average'].iloc[0]
    anio_estreno = int(filmacion['release_year'].iloc[0])

    # Verificar la condición de al menos 2000 valoraciones
    if votos < 2000:
        return {"mensaje": "La pelicula no cumple con al menos 2000 valoraciones"}

    mensaje = f"La pelicula {titulo} fue estrenada en el año {anio_estreno}.La misma cuenta con un total de {votos} votos y un valor promedio de {promedio_votos}."
    return {"mensaje": mensaje}

def calculate_return(movie_id):
    # Obtener el retorno de una película del DataFrame "dataset"
    movie_row = dataset[dataset['id'] == movie_id]
    return movie_row['revenue'].values[0] if not movie_row.empty else 0

def actor_success(actor_name, dataset2, dataset):
    
    # Filtrar el DataFrame por el nombre exacto del actor
    actor_movies = dataset2[dataset2['cast'].apply(lambda cast: any(actor_name.lower() in d['name'].lower() for d in ast.literal_eval(cast)))]


    movie_count = len(actor_movies)
    total_return = sum(calculate_return(movie_id) for movie_id in actor_movies['id'])
    average_return = total_return / movie_count if movie_count > 0 else 0
    return movie_count, total_return, average_return

@app.get("/get_actor/{actor_name}")
def get_actor_success(actor_name: str):
    count, total_return, average_return = actor_success(actor_name, dataset2, dataset)
    return {
        "message": f"El actor {actor_name} ha participado en {count} películas. Ha obtenido un retorno total de {total_return} con un promedio de retorno de {average_return} por película."
    }



@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    director_movies = []
    director_column = "Director"
    
    for index, row in dataset2.iterrows():
        cast = ast.literal_eval(row["cast"])
        crew = ast.literal_eval(row["crew"])
        
        for member in crew:
            if member["job"] == "Director" and member["name"] == nombre_director:
                movie_id = row["id"]
                movie = dataset.loc[dataset["id"] == movie_id]
                
                if not movie.empty:
                    movie_data = {
                        "Nombre": movie["title"].values[0],
                        "Fecha de lanzamiento": movie["release_date"].values[0],
                        "Retorno": movie["revenue"].values[0],
                        "Costo": movie["budget"].values[0],
                        "Ganancia": int(movie["revenue"].values[0]) - int(movie["budget"].values[0])
                    }
                    director_movies.append(movie_data)
    
    if director_movies:
        response = "Peliculas:\n\n"
        for i, movie in enumerate(director_movies):
            response += f"- Película {i+1}: {movie['Nombre']}\n  Fecha de estreno: {movie['Fecha de lanzamiento']}\n  Retorno: {movie['Retorno']}\n  Costo: {movie['Costo']}\n  Ganancia: {movie['Ganancia']}\n\n"
        
        return response
    else:
        return "El director no se encuentra en el dataset."
    
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(dataset["overview"])

# Función para obtener recomendaciones de películas
def get_movie_recommendations(movie_title, num_recommendations=5):
    # Buscar el índice de la película en el conjunto de datos
    movie_index = dataset[dataset["title"] == movie_title].index[0]

    # Calcular la similitud de coseno entre la película y todas las demás películas
    cosine_similarities = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()

    # Obtener los índices de las películas más similares
    similar_movie_indices = cosine_similarities.argsort()[:-num_recommendations-1:-1]

    # Obtener los títulos de las películas recomendadas
    recommended_movies = dataset.loc[similar_movie_indices, "title"].tolist()

    return recommended_movies

@app.get("/recomendacion/{movie_title}")
def get_recommendations(movie_title: str):
    recommendations = get_movie_recommendations(movie_title)
    return {"movie_title": movie_title, "recommendations": recommendations}
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


