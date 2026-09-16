import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


class FraudDataset(Dataset):
    def __init__(self, X, y, scaler=None, fit_scaler=False):
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        if scaler is not None:
            if fit_scaler:
                X_np = scaler.fit_transform(X_np)
            else:
                X_np = scaler.transform(X_np)

        self.X = torch.tensor(X_np, dtype=torch.float32)
        self.y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_train, X_test, y_train, y_test, batch_size=256, use_scaler=True):
    scaler = StandardScaler() if use_scaler else None

    train_ds = FraudDataset(
        X_train, y_train, scaler=scaler, fit_scaler=use_scaler
    )
    test_ds = FraudDataset(
        X_test, y_test, scaler=scaler, fit_scaler=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler