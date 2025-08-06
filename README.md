# Churn & Dropout Prediction Models and APIs

This repository hosts two proof of concept machine learning models and apis built to predict customer churn and student dropout risk. Each model is trained in reproducible, explainable pipelines using scikit-learn and deployed to **Google Cloud Run** via **Docker**, **Terraform**, and **GitHub Actions**.

---

## 📚 Dataset Sources

- **Dropout Prediction**: Based on the [Predict Students' Dropout and Academic Success dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) from the UCI Machine Learning Repository. This dataset includes demographic, academic, and socioeconomic data from a Portuguese higher education institution.

- **Churn Prediction**: Based on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download) from Kaggle. It contains customer demographics, service usage, and account information from a fictional telecom provider.

## 🚀 Live APIs

| Service        | Swagger                  | Method | Purpose                                   |
|----------------|--------------------------|--------|-------------------------------------------|
| `churn-api`    | `/predict`               | POST   | Predicts if a telecom customer will churn |
| `dropout-api`  | `/predict`               | POST   | Predicts if a student will drop out       |

You can try out the apis via swagger here: 

- Churn: https://churn-api-204458003569.europe-west2.run.app/docs
- Dropout: https://dropout-api-204458003569.europe-west2.run.app/docs

---

## 📦 Sample Payloads

### 🔹 Churn Prediction

To understand what the numbers mean, as well what the different options are in the payloads below, follow the dataset links above.

```json
{
  "gender": "Male",
  "seniorcitizen": 0,
  "partner": "Yes",
  "dependents": "Yes",
  "tenure": 12,
  "phoneservice": "Yes",
  "multiplelines": "Yes",
  "internetservice": "Fiber optic",
  "onlinesecurity": "No",
  "onlinebackup": "Yes",
  "deviceprotection": "Yes",
  "techsupport": "Yes",
  "streamingtv": "No",
  "streamingmovies": "No",
  "contract": "Month-to-month",
  "paperlessbilling": "Yes",
  "paymentmethod": "Electronic check",
  "monthlycharges": 30.00,
  "totalcharges": "80.00"
}
```

### 🔹 Dropout Prediction

```json
{
  "marital_status": 1,
  "application_mode": 1,
  "application_order": 1,
  "course": 33,
  "daytime_evening_attendance": 1,
  "previous_qualification": 1,
  "previous_qualification_grade": 70,
  "nacionality": 11,
  "mothers_qualification": 4,
  "fathers_qualification": 4,
  "mothers_occupation": 123,
  "fathers_occupation": 123,
  "admission_grade": 100,
  "displaced": 0,
  "educational_special_needs": 0,
  "debtor": 0,
  "tuition_fees_up_to_date": 1,
  "gender": 1,
  "scholarship_holder": 1,
  "age_at_enrollment": 18,
  "international": 0,
  "curricular_units_1st_sem_credited": 2,
  "curricular_units_1st_sem_enrolled": 2,
  "curricular_units_1st_sem_evaluations": 2,
  "curricular_units_1st_sem_approved": 2,
  "curricular_units_1st_sem_grade": 70,
  "curricular_units_1st_sem_without_evaluations": 0,
  "curricular_units_2nd_sem_credited": 2,
  "curricular_units_2nd_sem_enrolled": 2,
  "curricular_units_2nd_sem_evaluations": 2,
  "curricular_units_2nd_sem_approved": 2,
  "curricular_units_2nd_sem_grade": 75,
  "curricular_units_2nd_sem_without_evaluations": 0,
  "unemployment_rate": 15,
  "inflation_rate": 8,
  "gdp": 3.5
}
```

---

## 🧠 ML Training & Pipelines

All models are built using reproducible pipelines that feature:

- ⚙️ Data cleaning and type detection
- ✅ Dynamic encoding (OneHotEncoder with `handle_unknown='ignore'`)
- 🧪 Model traning

Each model and encoder is trained using `scikit-learn` and saved as `.pkl` files in `machine_learning/models` and then ported over to the `machine-learning-assets` folder in each api project.

---

## 🐳 Containerization & Deployment

### 🔧 Docker

Each service has a dedicated Dockerfile that:
- Installs dependencies via `requirements.txt`
- Loads models and encoders
- Launches via `uvicorn` on the correct port

### ☁️ GCP Cloud Run

Deployment is powered by:
- **Docker → Artifact Registry**
- **Terraform → Cloud Run**
- **GitHub Actions → CI/CD trigger**


## ✅ To Run Locally

```bash
# Create venv
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

# Install
pip install -r requirements.txt
```

### Running the APIs

```bash
uvicorn churn_api.app.main:app --reload --port 8080
uvicorn dropout_api.app.main:app --reload --port 8081
```

### Training models and encoders

```bash
python -m machine_learning.training.domain.churn.train_churn_v1_model
python -m machine_learning.training.domain.dropout.train_dropout_v1_model
```
