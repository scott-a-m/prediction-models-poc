from fastapi import APIRouter, Request
from dropout_api.app.schema import StudentInput
from dropout_api.app.prediction_pipeline import preprocess_data_and_predict
from dropout_api.app.assets_loader import load_model, load_encoders
from dropout_api.app.rate_limit_config import limiter

router = APIRouter()
model = load_model("dropout_model_v1.pkl", "machine-learning-assets/models")
encoders = load_encoders("dropout_encoders.pkl", "machine-learning-assets/encoders")

@router.post("/predict")
@limiter.limit("15/minute")
def predict_dropout(request: Request, data: StudentInput):
    input_dict = data.model_dump()
    return preprocess_data_and_predict(model, encoders, input_dict)