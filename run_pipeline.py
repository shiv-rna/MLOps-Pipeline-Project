from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    # Run pipeline
    train_pipeline(data_path="/workspaces/MLOps-Pipeline-Project/data/olist_customers_dataset.csv")
