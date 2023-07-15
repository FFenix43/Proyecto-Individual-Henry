# Proyecto-Individual-Henry

En este repositorio van a encontrar los codigos para transformar los archivos excel, el modelo de EDA y las APIS que se piden.

Funciona todos los codigos que estan.

El RENDER, me tira "out of memory" por eso no lo puedo levantar.

Breve explicacion de las API:
  Esta aplicación se encarga de analizar y manipular datos de películas. En primer lugar, se importan varias bibliotecas como FastAPI, pandas, datetime, nltk, entre otras. A continuación, se cargan dos conjuntos de datos de películas desde archivos CSV en los DataFrames 'movies' y 'credits'. Luego, se realizan algunas operaciones de limpieza y preprocesamiento de datos en el DataFrame 'dataset', como eliminar filas con valores faltantes y rellenar valores nulos en la columna 'overview'. A continuación, se definen varias funciones que se utilizan como controladores de ruta para manejar las solicitudes HTTP. Estas funciones permiten obtener información sobre la cantidad de películas estrenadas en un mes o día de la semana específico, el puntaje y los votos de una película, el éxito de un actor en términos de la cantidad de películas y retorno financiero, y también proporcionan recomendaciones de películas similares a una película dada. Finalmente, el servidor FastAPI se inicia y se ejecuta en el puerto 8000. En general, esta aplicación web ofrece una forma conveniente de acceder y analizar datos relacionados con películas.
  

