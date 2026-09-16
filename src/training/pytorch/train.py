import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class Trainer:
        def __init__(
                        self, 
                        loss_fn, 
                        optimizer, 
        ):
                self.loss_fn = loss_fn
                self.optimizer = optimizer

        def train_loop(self, dataloader, model):
                size = len(dataloader.dataset)
                model.train()
                for idx, (X_batch, y_batch) in enumerate(dataloader):
                        preds = model(X_batch)
                        loss = self.loss_fn(preds, y_batch)

                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        if idx % 100 == 0:
                                loss, current = loss.item(), idx * len(X_batch) + len(X_batch)
                                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        def test_loop(self, dataloader, model):
                model.eval()
                num_batches = len(dataloader)
                total_loss = 0.0
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    for idx, (X_batch, y_batch) in enumerate(dataloader):
                        preds = model(X_batch)
                        loss = self.loss_fn(preds, y_batch)
                        total_loss += loss.item()
                        
                        all_preds.append(preds.cpu().numpy())
                        all_targets.append(y_batch.cpu().numpy())
                
                avg_loss = total_loss / num_batches
                
                all_preds = np.concatenate(all_preds).flatten()
                all_targets = np.concatenate(all_targets).flatten()
                
                y_pred_binary = (all_preds >= model.threshold).astype(int)
                
                acc = accuracy_score(all_targets, y_pred_binary)
                prec = precision_score(all_targets, y_pred_binary, zero_division=0)
                recall = recall_score(all_targets, y_pred_binary, zero_division=0)
                f1 = f1_score(all_targets, y_pred_binary, zero_division=0)
                roc_auc = roc_auc_score(all_targets, y_pred_binary)
                
                print(f"\nTest Results:")
                print(f"Loss: {avg_loss:>7f}")
                print(f"Accuracy: {acc:>7f}")
                print(f"Precision: {prec:>7f}")
                print(f"Recall: {recall:>7f}")
                print(f"F1: {f1:>7f}")
                print(f"ROC-AUC: {roc_auc:>7f}")
                
                return {
                        "loss": avg_loss,
                        "accuracy": acc,
                        "precision": prec,
                        "recall": recall,
                        "f1": f1,
                        "roc_auc": roc_auc
                }




