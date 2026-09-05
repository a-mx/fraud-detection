import argparse
import torch
from torch import nn, optim
from src.data.build_dataset import build_dataset
from src.models.baseline import BaselineModel
from src.models.mlp import MLP
from src.training.train import Trainer
from src.training.dataloader import make_loaders
from src.config.settings import Settings

def main():
    parser = argparse.ArgumentParser(description="Fraud detection model training")
    parser.add_argument(
        "--model", 
        type=str, 
        default="baseline",
        choices=["baseline", "mlp"],
        help="Which model to train"
    )
    
    args = parser.parse_args()
    settings = Settings.from_yaml()
    
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = build_dataset()
    
    if args.model == "baseline":
        print(f"Training {args.model}...")
        model = BaselineModel(
            max_iter=settings.models.baseline.max_iter,
            class_weight=settings.models.baseline.class_weight
        )
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        print(f"Results: {metrics}")
    
    elif args.model == "mlp":
        print(f"Training {args.model}...")
        input_size = X_train.shape[1]
        model = MLP(
            input_size=input_size,
            hidden_size=settings.models.mlp.hidden_size,
            output_size=settings.models.mlp.output_size,
            threshold=settings.models.mlp.threshold
        )
        
        train_loader, test_loader = make_loaders(
            X_train, X_test, y_train, y_test, 
            batch_size=settings.models.mlp.batch_size
        )
        
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=settings.models.mlp.lr)
        trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)
        
        for epoch in range(settings.models.mlp.epochs):
            print(f"\nEpoch {epoch + 1}/{settings.models.mlp.epochs}")
            trainer.train_loop(train_loader, model)
            metrics = trainer.test_loop(test_loader, model)

if __name__ == "__main__":
    main()