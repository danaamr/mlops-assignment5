import mlflow
import sys
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))

with open("model_info.txt") as f:
    run_id = f.read().strip()

print("Run ID:", run_id)
print("Checking model performance")

accuracy = float(os.getenv("ACCURACY", 0.8))

print("Accuracy:", accuracy)

if accuracy < 0.85:
    print("Failed")
    sys.exit(1)
else:
    print("Passed")