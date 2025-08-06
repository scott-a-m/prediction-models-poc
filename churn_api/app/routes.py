from fastapi import APIRouter, Request
from churn_api.app.schema import CustomerInput
from churn_api.app.prediction_pipeline import preprocess_data_and_predict
from churn_api.app.assets_loader import load_model, load_encoders
from churn_api.app.rate_limit_config import limiter

router = APIRouter()
model = load_model("churn_model_v1.pkl", "machine-learning-assets/models")
encoders = load_encoders("churn_encoders.pkl", "machine-learning-assets/encoders")

@router.post("/predict")
@limiter.limit("15/minute")
def predict_churn(request: Request, data: CustomerInput):
    input_dict = data.model_dump()
    return preprocess_data_and_predict(model, encoders, input_dict)