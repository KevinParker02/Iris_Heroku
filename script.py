from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
import os

app = FastAPI()

# Configurar rutas de templates y archivos estáticos
templates = Jinja2Templates(directory="templates")

# Función de predicción
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    loaded_model = pickle.load(open("checkpoints/model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

# Ruta index
@app.get("/", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ruta resultado
@app.post("/result", response_class=HTMLResponse)
async def result(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    to_predict_list = [sepal_length, sepal_width, petal_length, petal_width]
    try:
        result = ValuePredictor(to_predict_list)
        if int(result) == 0:
            prediction = 'Iris-Setosa'
        elif int(result) == 1:
            prediction = 'Iris-Virginica'
        elif int(result) == 2:
            prediction = 'Iris-Versicolour'
        else:
            prediction = f'{int(result)} No-definida'
    except:
        prediction = 'Error en el formato de los datos'

    return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction})
