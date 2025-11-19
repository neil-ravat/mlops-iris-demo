from pycaret.classification import setup, compare_models
from pycaret.datasets import get_data
import joblib

# Load iris dataset
iris = get_data('iris')

# Setup PyCaret experiment
exp = setup(
    data=iris,
    target='species',
    session_id=42,
    silent=True,
    html=False
)

# Train and select best model
best_model = compare_models()

# Save model
joblib.dump(best_model, 'iris_model.pkl')

print("Model trained and saved as iris_model.pkl")
