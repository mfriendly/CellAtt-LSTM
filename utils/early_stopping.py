import os
import torch


class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, min_epochs=3):
        self.patience = patience
        self.delta = delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.current_epoch = 0

    def __call__(self, val_loss, model, save_path):
        self.current_epoch += 1
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, save_path)
        elif score < self.best_score + self.delta:
            if self.current_epoch >= self.min_epochs:
                self.counter += 1
                print(f"EarlyStopping: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(model, save_path)
            self.counter = 0

    def _save_checkpoint(self, model, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))
