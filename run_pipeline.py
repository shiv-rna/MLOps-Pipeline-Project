from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # Run pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/workspaces/MLOps-Pipeline-Project/data/olist_customers_dataset.csv")

# Use the URI printed and run : mlflow ui --backend-store-uri file:/home/codespace/.config/zenml/local_stores/b59e27b5-62a7-4ac7-9291-ebc3f13ffaf1/mlruns
