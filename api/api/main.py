import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
#from processing.pipeline import financial_data_pipeline


app = FastAPI(
    title="Eurostoxx50 forecasting",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model = pickle.load(
    open('model.pkl', 'rb')
)


@app.get("/")
def read_root(text: str = ""):
    if not text:
        return f"Try to append ?text=something in the URL!"
    else:
        return text


class TradingDay(BaseModel):
    Open : float
    High : float
    Low : float
    Close : float
    Adj_Close : float
    Volume : float
    BBL_20_2: float
    BBM_20_2: float 
    BBU_20_2: float
    BBB_20_2: float
    BBP_20_2: float
    RSI: float
    Month_sin: float
    Month_cos: float
    Weekday_sin: float
    Weekday_cos: float
    Date_numeric: float
    Close_t1: float

@app.post("/predict/")
def predict(trading_data: List[TradingDay]) -> float:
    X = np.array([dict(trading_day) for trading_day in trading_data])

    #X_preprocessed = financial_data_pipeline.fit_transform(X)

    y_pred = model.predict(X)
    
    return list(y_pred)
