from src.data.build_dataset import build_dataset
from src.models.baseline import train_baseline, evaluate_baseline

def main():
    X_train, X_test, y_train, y_test = build_dataset()
    model = train_baseline(X_train, y_train)
    metrics = evaluate_baseline(model, X_test, y_test)

    print(metrics)

if __name__ == "__main__":
    main()