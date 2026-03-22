import mlflow
import sys
import os

# use secret if available
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))

with open("model_info.txt") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0)

print("Accuracy:", accuracy)

if accuracy < 0.85:
    print("Failed")
    sys.exit(1)
else:
    print("Passed")