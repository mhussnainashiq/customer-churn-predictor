from app.ml import train
if __name__ == "__main__":
    _, metrics = train()
    print("Model trained and saved. Metrics are for demonstration data, not real-world performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
