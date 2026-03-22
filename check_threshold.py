import mlflow
import sys

mlflow.set_tracking_uri("file:./mlruns")

with open("model_info.txt") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
accuracy = run.data.metrics["accuracy"]

print("Accuracy:", accuracy)

if accuracy < 0.85:
    print("Failed")
    sys.exit(1)
else:
    print("Passed")