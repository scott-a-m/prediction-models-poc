from fastapi import APIRouter
from dropout_api.app.schema import StudentInput
from dropout_api.app.prediction_pipeline import preprocess_data_and_predict
from dropout_api.app.assets_loader import load_model, load_encoders

router = APIRouter()
model = load_model("dropout_model_v1.pkl", "machine-learning-assets/models")
encoders = load_encoders("dropout_encoders.pkl", "machine-learning-assets/encoders")

@router.post("/predict")
def predict_dropout(data: StudentInput):
    input_dict = data.model_dump()
    return preprocess_data_and_predict(model, encoders, input_dict)

# {
#   "marital_status": 1,
#   "application_mode": 1,
#   "application_order": 1,
#   "course": 33,
#   "daytime_evening_attendance": 1,
#   "previous_qualification": 1,
#   "previous_qualification_grade": 70,
#   "nacionality": 11,
#   "mothers_qualification": 4,
#   "fathers_qualification": 4,
#   "mothers_occupation": 123,
#   "fathers_occupation": 123,
#   "admission_grade": 100,
#   "displaced": 0,
#   "educational_special_needs": 0,
#   "debtor": 0,
#   "tuition_fees_up_to_date": 1,
#   "gender": 1,
#   "scholarship_holder": 1,
#   "age_at_enrollment": 18,
#   "international": 0,
#   "curricular_units_1st_sem_credited": 2,
#   "curricular_units_1st_sem_enrolled": 2,
#   "curricular_units_1st_sem_evaluations": 2,
#   "curricular_units_1st_sem_approved": 2,
#   "curricular_units_1st_sem_grade": 70,
#   "curricular_units_1st_sem_without_evaluations": 0,
#   "curricular_units_2nd_sem_credited": 2,
#   "curricular_units_2nd_sem_enrolled": 2,
#   "curricular_units_2nd_sem_evaluations": 2,
#   "curricular_units_2nd_sem_approved": 2,
#   "curricular_units_2nd_sem_grade": 75,
#   "curricular_units_2nd_sem_without_evaluations": 0,
#   "unemployment_rate": 15,
#   "inflation_rate": 8,
#   "gdp": 3.5
# }