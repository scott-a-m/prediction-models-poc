from fastapi import FastAPI
from  dropout_api.app.routes import router

app = FastAPI(title="Student Dropout Prediction API")
app.include_router(router)