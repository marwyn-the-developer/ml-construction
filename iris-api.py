# SID DDA
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from skops.io import load
import numpy as np

app = FastAPI(
    title="Iris API",
    description="classifying plants"
)
iris_classifier = load('iris_classifier.skops', trusted=True)

class FlowerModel(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get('/')
async def index():
    return {"Message": "This is Index"}

@app.post('/species/predictions')
async def prediction(flower: FlowerModel):
    flower_sample = np.array([flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]).reshape(1, -1)
    prediction = iris_classifier.predict(flower_sample)
    return {'species': prediction[0]}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

