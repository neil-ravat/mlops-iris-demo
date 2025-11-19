from pycaret.classification import *
from pycaret.datasets import get_data
import joblib

# Load training data
iris = get_data('iris')

# PyCaret setup
exp = setup(
    data=iris,
    target='species',
    session_id=42,
    html=False
)

# Compare all models
best_model = compare_models()

# Finalize (fit on full data)
final_model = finalize_model(best_model)

# Save the updated model
joblib.dump(final_model, 'iris_model.pkl')

print("New model retrained successfully and saved as iris_model.pkl")
