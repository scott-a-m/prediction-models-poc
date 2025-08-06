from fastapi import FastAPI
from  dropout_api.app.routes import router
from churn_api.app.rate_limit_config import limiter, rate_limit_handler
from slowapi.errors import RateLimitExceeded

app = FastAPI(title="Student Dropout Prediction API")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

app.include_router(router)