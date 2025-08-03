from fastapi import APIRouter
from churn_api.app.schema import CustomerInput
from churn_api.app.prediction_pipeline import preprocess_data_and_predict
from churn_api.app.assets_loader import load_model, load_encoders

router = APIRouter()
model = load_model("churn_model_v1.pkl", "machine-learning-assets/models")
encoders = load_encoders("churn_encoders.pkl", "machine-learning-assets/encoders")

@router.post("/predict")
def predict_churn(data: CustomerInput):
    input_dict = data.model_dump()
    return preprocess_data_and_predict(model, encoders, input_dict)

# {
#   "gender": "Male",
#   "seniorcitizen": 0,
#   "partner": "Yes",
#   "dependents": "Yes",
#   "tenure": 12,
#   "phoneservice": "Yes",
#   "multiplelines": "Yes",
#   "internetservice": "Fiber optic",
#   "onlinesecurity": "No",
#   "onlinebackup": "Yes",
#   "deviceprotection": "Yes",
#   "techsupport": "Yes",
#   "streamingtv": "No",
#   "streamingmovies": "No",
#   "contract": "Month-to-month",
#   "paperlessbilling": "Yes",
#   "paymentmethod": "Electronic check",
#   "monthlycharges": 30.00,
#   "totalcharges": "80.00"
# }