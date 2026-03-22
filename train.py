import mlflow

# use GitHub secret later
mlflow.set_tracking_uri("file:./mlruns")

accuracy = 0.9  # change later

print("Training model...")

with mlflow.start_run() as run:
    run_id = run.info.run_id

    mlflow.log_metric("accuracy", accuracy)

    # save run_id (NOT accuracy now)
    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print("Run ID:", run_id)
    print("Accuracy logged:", accuracy)