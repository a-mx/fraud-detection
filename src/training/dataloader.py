import torch
from torch.utils.data import DataLoader, TensorDataset

def make_loaders(X_train, X_test, y_train, y_test, batch_size=256):
    train_ds = TensorDataset(
        torch.tensor(X_train.to_numpy(), dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
    )

    test_ds = TensorDataset(
        torch.tensor(X_test.to_numpy(), dtype=torch.float32),
        torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader