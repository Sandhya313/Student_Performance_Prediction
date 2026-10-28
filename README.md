# Student Placement Prediction API

A production-ready FastAPI service and MLOps deployment pipeline for student placement prediction using a pre-trained scikit-learn machine learning pipeline.

---

## 📌 Project Overview

This repository provides an automated backend API and MLOps deployment pipeline serving a pre-trained model (`model.joblib`) created in Google Colab. The model predicts student placement status (`placement_status`) based on six academic and behavioural indicators.

### 🛡️ Preventing Target Leakage
The feature `exam_score` was explicitly **excluded** during training to prevent target leakage. The API strictly forbids and rejects requests containing `exam_score` using Pydantic validation (`extra="forbid"`), returning an HTTP `422 Unprocessable Entity` response.

### 📊 Model Specifications
- **Model File**: `model.joblib` (~26.1 MB)
- **Model Type**: Scikit-Learn Pipeline (`ColumnTransformer` with `SimpleImputer` + `RandomForestClassifier`)
- **Trained With**: `scikit-learn==1.6.1`
- **Output Classes**: `["Not Placed", "Placed"]`
- **Prediction Capabilities**: Class label (`placement_status`) and class confidence probabilities (`probabilities`).

### 📥 Expected Input Features (Exact 6)
| Feature Name | Data Type | Description | Valid Range |
| :--- | :--- | :--- | :--- |
| `study_hours` | `float` | Daily study hours | $\ge 0$ |
| `attendance` | `float` | Class attendance percentage | $0 - 100$ |
| `sleep_hours` | `float` | Daily sleep hours | $\ge 0$ |
| `internet_usage` | `float` | Daily internet usage hours | $\ge 0$ |
| `assignments_completed` | `float` | Number of completed assignments | $\ge 0$ |
| `previous_score` | `float` | Prior academic exam/aggregate score | $0 - 100$ |

---

## 🗂️ Repository Structure

```text
student/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline (Tests + Docker smoke test)
├── tests/
│   └── test_api.py              # Automated test suite using pytest & TestClient
├── .dockerignore                # Excludes virtual environments and temp files from Docker
├── .gitignore                   # Excludes caches, venvs, and editor configs
├── Dockerfile                   # Production-grade multi-stage container specification
├── main.py                      # FastAPI application with routes and Pydantic validation
├── model.joblib                 # Trained ML pipeline artifact (preserved as-is)
├── README.md                    # Comprehensive technical documentation
└── requirements.txt             # Pinned production dependencies (scikit-learn==1.6.1)
```

---

## ⚙️ Local Installation & Setup

### 1. Prerequisites
- **Python**: Version `3.11` recommended for scikit-learn 1.6.1 binary wheel compatibility.
- **Git**
- **Docker** (optional for local container runs)

### 2. Create Virtual Environment
Using Python's built-in `venv`:
```bash
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate
```

Alternatively, using `uv`:
```bash
uv venv --python 3.11 .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the API Locally
Start Uvicorn server:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Once started:
- API Root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Alternative ReDoc UI: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🔌 API Endpoints & Usage

### 1. Welcome Endpoint (`GET /`)
Returns general service information and navigation links.
```bash
curl http://127.0.0.1:8000/
```
**Response (200 OK):**
```json
{
  "message": "Welcome to the Student Placement Prediction API",
  "docs_url": "/docs",
  "health_url": "/health",
  "version": "1.0.0"
}
```

---

### 2. Health Check (`GET /health`)
Verifies service liveness and confirms the ML model is loaded and operational.
```bash
curl http://127.0.0.1:8000/health
```
**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "expected_features": [
    "study_hours",
    "attendance",
    "sleep_hours",
    "internet_usage",
    "assignments_completed",
    "previous_score"
  ],
  "error": null
}
```

---

### 3. Placement Prediction (`POST /predict`)
Predicts student placement status given the six required features.

#### Example Request (cURL)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "study_hours": 6.5,
    "attendance": 90.0,
    "sleep_hours": 7.0,
    "internet_usage": 2.0,
    "assignments_completed": 9.0,
    "previous_score": 85.0
  }'
```

#### Example Request (PowerShell)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"study_hours": 6.5, "attendance": 90.0, "sleep_hours": 7.0, "internet_usage": 2.0, "assignments_completed": 9.0, "previous_score": 85.0}'
```

#### Successful Response (200 OK)
```json
{
  "placement_status": "Placed",
  "probabilities": {
    "Not Placed": 0.025,
    "Placed": 0.975
  },
  "model_version": "1.0.0"
}
```

---

### 4. Error Handling & Validation

#### Case A: Target Leakage Attempt (Passing `exam_score`)
Sending `exam_score` triggers strict rejection:
```json
{
  "study_hours": 5.0,
  "attendance": 85.0,
  "sleep_hours": 7.0,
  "internet_usage": 3.0,
  "assignments_completed": 8.0,
  "previous_score": 75.0,
  "exam_score": 92.0
}
```
**Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "type": "extra_forbidden",
      "loc": ["body", "exam_score"],
      "msg": "Extra inputs are not permitted",
      "input": 92.0
    }
  ]
}
```

#### Case B: Out-of-Range or Missing Fields
If required fields are missing or values violate constraints (e.g., `attendance > 100` or negative hours), Pydantic returns HTTP `422` with granular error locations.

---

## 🐳 Docker Deployment

The application includes an optimized, non-root `Dockerfile` based on `python:3.11-slim` with built-in healthchecks.

### 1. Build Docker Image
```bash
docker build -t student-placement-api .
```

### 2. Run Docker Container
```bash
docker run --rm --name student-placement-container -p 8000:8000 student-placement-api
```

### 3. Verify Container Health
```bash
curl http://localhost:8000/health
```

### 4. Stop Container
```bash
docker stop student-placement-container
```

---

## 🧪 Testing Suite

Automated tests are implemented in `tests/test_api.py` using `pytest` and FastAPI's `TestClient`. All tests execute against the actual saved model (`model.joblib`), ensuring no mocked behavior.

### Run Tests
```bash
pytest tests/test_api.py -v
```

### Covered Test Cases
1. `test_home_endpoint`: Confirms root endpoint availability and response schema.
2. `test_health_endpoint`: Asserts model load status and expected 6-feature list.
3. `test_valid_prediction_request`: Tests prediction with typical student input.
4. `test_prediction_response_format`: Validates probability distributions and status consistency.
5. `test_invalid_input_missing_field`: Asserts HTTP 422 upon missing required feature.
6. `test_invalid_input_wrong_type`: Asserts HTTP 422 on string-in-place-of-float.
7. `test_invalid_input_out_of_range`: Asserts HTTP 422 when percentage exceeds 100.
8. `test_rejection_of_exam_score_feature`: Asserts explicit rejection of `exam_score` to prevent target leakage.

---

## 🚀 GitHub Actions CI/CD Pipeline

The `.github/workflows/ci.yml` pipeline runs automatically on `push` and `pull_request` against `main` and `master`:

1. **Lint & Test Suite**:
   - Sets up Python 3.11.
   - Verifies all required artifacts exist (`model.joblib`, `main.py`, `requirements.txt`, `Dockerfile`, `tests/test_api.py`).
   - Installs dependencies.
   - Executes the full `pytest` suite.
2. **Build & Smoke-Test Docker Image**:
   - Builds the Docker image without requiring external Docker Hub credentials.
   - Starts a container in detached mode.
   - Performs a curl health check against `http://localhost:8000/health`.
   - Shuts down the container safely.

---

## 📦 Model Artifact Storage & MLOps Considerations

### GitHub File Size Limits
- Current `model.joblib` size: **~26.1 MB**.
- GitHub warns on files **> 50 MB** and blocks pushes for files **> 100 MB**.
- Since `model.joblib` is ~26.1 MB, it can be directly tracked in Git for testing and assignment evaluation.

### Production Recommendations for Larger Models
If the model size expands in future iterations (> 50 MB):
1. **Git LFS (Large File Storage)**: Track `.joblib` files with Git LFS (`git lfs track "*.joblib"`).
2. **Cloud Object Storage (S3 / GCS / Azure Blob)**: Store models in versioned object buckets and download them during container startup via an initialization script or S3 URI.
3. **MLflow / Weights & Biases Model Registry**: Manage model artifacts with semantic versioning, lineage tracking, and automated deployment webhooks.
4. **Hugging Face Model Hub**: Version public or private models with automated Git-backed endpoints.

---

## 📋 Instructions to Create & Push to GitHub

To publish this project to your GitHub account:

### Step 1: Initialize Git Locally
```bash
git init
git add .
git commit -m "feat: complete ML + MLOps student placement API with tests, Docker, and CI"
```

### Step 2: Create a New GitHub Repository
You can create the repository either via the GitHub Web UI or using the GitHub CLI (`gh`):

**Option A: Using GitHub CLI (recommended)**:
```bash
# Authenticate if not already logged in
gh auth login

# Create a public or private repository and push immediately
gh repo create student-placement-api --public --source=. --remote=origin --push
```

**Option B: Using the GitHub Web Interface**:
1. Go to [github.com/new](https://github.com/new).
2. Name your repository: `student-placement-api`.
3. Leave "Initialize this repository with a README" **unchecked**.
4. Click **Create repository**.
5. Link and push your local repository:
```bash
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/student-placement-api.git
git push -u origin main
```
