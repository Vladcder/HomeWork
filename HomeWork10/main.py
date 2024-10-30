import json

import  joblib
import  uvicorn
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import  JSONResponse
from numpy import int8, int16
from pydantic import BaseModel, ConfigDict, Field
import  sklearn

app=FastAPI()

with open('realty_model.pkl','rb') as file:
    model=joblib.load(file)

class ModelRequestData(BaseModel):
    lat:float=Field(gt=55.468426, lt=56.028824)
    lon:float=Field(gt=37.136489, lt=38.122467)
    total_square: float=Field(gt=5, lt=5000)
    rooms:int=Field(gt=0, lt=30)
    floor: int=Field(gt=0, lt=100)


class Result(BaseModel):
    result: float




@app.get('/health')
def health():
    return  JSONResponse(content={'message': 'It is alive!'},status_code=200)

@app.get('/predict_get', response_model=Result)
def preprocess_data_get(data:ModelRequestData= Query(None)):
    input_data = data.model_dump()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)
    return Result(result=result)



@app.post('/predict_post', response_model=Result)
def preprocess_data(data:ModelRequestData):
    input_data=data.model_dump()
    input_df=pd.DataFrame(input_data, index=[0])
    result=model.predict(input_df)[0]
    return  Result(result=result)

if __name__=='__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

