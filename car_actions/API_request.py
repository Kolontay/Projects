import pandas as pd
import joblib
import dill as dill
from typing import Optional
import numpy as np




from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


# Загружаем обработку
with open('./models/preprocessor.pkl', 'rb') as f:
   preprocessor = dill.load(f)


# Загрузка модель
filename = './models/cars_subscribe_model_1.joblib'
model_car = joblib.load(filename)




class Form(BaseModel):
   session_id: str
   client_id: str
   visit_date: Optional[str]
   visit_time: Optional[str]
   visit_number: int
   utm_source: Optional[str]
   utm_medium: Optional[str]
   utm_campaign: Optional[str]
   utm_adcontent: Optional[str]
   utm_keyword: Optional[str]
   device_category: Optional[str]
   device_os: Optional[str]
   device_brand: Optional[str]
   device_model: Optional[str]
   device_screen_resolution: Optional[str]
   device_browser: Optional[str]
   geo_country: Optional[str]
   geo_city: Optional[str]




class Prediction(BaseModel):
   client_id: str
   Result: int




@app.get('/status')
def status():
   return "OK"




@app.get('/version')
def version():
   return model_car['info']




@app.post('/predict', response_model=Prediction)
def predict(form: Form):
   df = pd.DataFrame.from_dict([form.dict()])
   test = preprocessor.transform(df).drop(['session_id', 'client_id'], axis=1)
   preds_proba = model_car['model'].predict_proba(test)[:, 1]
   preds = (preds_proba >= 0.04).astype(int)


   return {
       'client_id': form.client_id,
       'Result': preds[0]
   }
