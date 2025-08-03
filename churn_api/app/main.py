from fastapi import FastAPI
from  churn_api.app.routes import router

app = FastAPI(title="Churn Prediction API")
app.include_router(router)