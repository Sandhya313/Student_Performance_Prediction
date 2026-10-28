import sys
from pathlib import Path

# Ensure project root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
import pytest
from main import app, model_pipeline

client = TestClient(app)


def test_home_endpoint():
    """Verify GET / returns 200 OK and expected welcome message structure."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Welcome" in data["message"]
    assert "docs_url" in data
    assert "health_url" in data


def test_health_endpoint():
    """Verify GET /health returns 200 OK and model is successfully loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert isinstance(data["expected_features"], list)
    assert len(data["expected_features"]) == 6
    assert "study_hours" in data["expected_features"]
    assert "exam_score" not in data["expected_features"]


def test_valid_prediction_request():
    """Verify POST /predict succeeds with valid features and actual saved model."""
    payload = {
        "study_hours": 6.5,
        "attendance": 90.0,
        "sleep_hours": 7.0,
        "internet_usage": 2.0,
        "assignments_completed": 9.0,
        "previous_score": 85.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "placement_status" in data
    assert data["placement_status"] in ["Placed", "Not Placed"]


def test_prediction_response_format():
    """Verify detailed response structure including probabilities from the actual model."""
    payload = {
        "study_hours": 5.0,
        "attendance": 80.0,
        "sleep_hours": 6.5,
        "internet_usage": 3.0,
        "assignments_completed": 7.0,
        "previous_score": 75.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Placement status check
    assert isinstance(data["placement_status"], str)
    assert data["placement_status"] in ["Placed", "Not Placed"]

    # Model version check
    assert data.get("model_version") == "1.0.0"

    # Probabilities format check
    assert "probabilities" in data
    probabilities = data["probabilities"]
    assert isinstance(probabilities, dict)
    assert "Placed" in probabilities
    assert "Not Placed" in probabilities

    # Probabilities value ranges and sum
    for cls_name, prob in probabilities.items():
        assert 0.0 <= prob <= 1.0

    prob_sum = sum(probabilities.values())
    assert pytest.approx(prob_sum, abs=0.02) == 1.0


def test_invalid_input_missing_field():
    """Verify POST /predict rejects payloads missing required features with 422."""
    payload = {
        "attendance": 80.0,
        "sleep_hours": 7.0,
        "internet_usage": 2.0,
        "assignments_completed": 8.0,
        "previous_score": 70.0,
        # missing study_hours
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_input_wrong_type():
    """Verify POST /predict rejects non-numeric input values with 422."""
    payload = {
        "study_hours": "many_hours",  # Invalid type
        "attendance": 85.0,
        "sleep_hours": 7.0,
        "internet_usage": 3.0,
        "assignments_completed": 8.0,
        "previous_score": 75.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_input_out_of_range():
    """Verify POST /predict rejects percentage out-of-range (>100 or <0) with 422."""
    payload = {
        "study_hours": 5.0,
        "attendance": 150.0,  # Attendance cannot exceed 100%
        "sleep_hours": 7.0,
        "internet_usage": 3.0,
        "assignments_completed": 8.0,
        "previous_score": 75.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_rejection_of_exam_score_feature():
    """
    CRITICAL: Verify POST /predict explicitly rejects requests containing exam_score
    to prevent target leakage.
    """
    payload_with_leakage = {
        "study_hours": 5.0,
        "attendance": 85.0,
        "sleep_hours": 7.0,
        "internet_usage": 3.0,
        "assignments_completed": 8.0,
        "previous_score": 75.0,
        "exam_score": 92.0,  # Must be strictly rejected
    }
    response = client.post("/predict", json=payload_with_leakage)
    assert response.status_code == 422

    # Verify that the validation error specifically cites exam_score
    errors = response.json().get("detail", [])
    extra_field_errors = [err for err in errors if "exam_score" in str(err.get("loc", []))]
    assert len(extra_field_errors) > 0, "Expected validation error indicating 'exam_score' is not allowed"
