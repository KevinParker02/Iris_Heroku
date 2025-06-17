from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
import os

# Cargar templates
templates = Jinja2Templates(directory="templates")

# Crear app
app = FastAPI()

# Cargar modelo
modelo = pickle.load(open("checkpoints/model.pkl", "rb"))

@app.get("/", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/result", response_class=HTMLResponse)
async def result(
    request: Request,
    feature1: float = Form(...),
    feature2: float = Form(...),
    feature3: float = Form(...),
    feature4: float = Form(...)
):
    try:
        datos = [feature1, feature2, feature3, feature4]
        pred = modelo.predict([np.array(datos)])[0]
        if int(pred) == 0:
            prediction = 'Iris-Setosa'
        elif int(pred) == 1:
            prediction = 'Iris-Virginica'
        elif int(pred) == 2:
            prediction = 'Iris-Versicolour'
        else:
            prediction = f'{int(pred)} No-definida'
    except ValueError:
        prediction = 'Error en el formato de los datos'

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": prediction
    })