"""
example.py — end-to-end demo using synthetic data.

Runs without real data files, W&B, or heavy dependencies (mamba, momentfm).
Only requires: torch, pytorch_lightning, pytorch_tcn, scikit-learn, numpy.

Simulates a cohort of subjects each with a sinusoidal actigraphy-like signal
and a known acrophase, then trains and evaluates each supported model type.
"""

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from TransformerEncoder import TransformerEncoder

torch.manual_seed(42)
np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────

N_SUBJECTS   = 80   # number of simulated subjects
DAYS         = 7    # days of recording per subject
SEQ_LEN      = DAYS * 24
INPUT_SIZE   = 3    # activity + sin(hour) + cos(hour)
OUTPUT_SIZE  = 2    # sin(acrophase), cos(acrophase)
BATCH_SIZE   = 16
MAX_EPOCHS   = 10


# ── Synthetic data generation ─────────────────────────────────────────────────

def generate_synthetic_data(n_subjects=N_SUBJECTS, seq_len=SEQ_LEN, noise=0.3):
    """
    Generate a synthetic actigraphy dataset.

    Each subject has:
      - A sinusoidal activity signal peaking at a random acrophase hour
      - Time-of-day sin/cos features appended as additional input channels
      - A circular (sin, cos) target encoding the acrophase

    Returns:
        X : (n_subjects, seq_len, 3)  — activity + sin/cos hour
        y : (n_subjects, 2)           — sin/cos acrophase
    """
    acrophases = torch.FloatTensor(n_subjects).uniform_(0, 24)  # hours

    hours = torch.arange(seq_len).float()  # (seq_len,)
    # Activity signal: cosine centred on each subject's acrophase + noise
    hour_grid = hours.unsqueeze(0).expand(n_subjects, -1)           # (N, T)
    acro_grid = acrophases.unsqueeze(1).expand(-1, seq_len)         # (N, T)
    activity  = torch.cos(2 * torch.pi * (hour_grid - acro_grid) / 24)
    activity  = activity + noise * torch.randn_like(activity)
    activity  = activity.unsqueeze(2)                               # (N, T, 1)

    # Time-of-day features (same for all subjects, start hour = 0)
    sin_hour = torch.sin(2 * torch.pi * hours / 24).unsqueeze(0).unsqueeze(2).expand(n_subjects, -1, -1)
    cos_hour = torch.cos(2 * torch.pi * hours / 24).unsqueeze(0).unsqueeze(2).expand(n_subjects, -1, -1)

    X = torch.cat([activity, sin_hour, cos_hour], dim=2).float()

    # Circular target encoding
    angle = 2 * torch.pi * acrophases / 24
    y = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1).float()

    return X, y


def make_splits(X, y, train_frac=0.6, val_frac=0.2):
    """Simple random subject-wise train/val/test split."""
    n = X.shape[0]
    idx = torch.randperm(n)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    def split(t): return t[train_idx], t[val_idx], t[test_idx]

    X_train, X_val, X_test = split(X)
    y_train, y_val, y_test = split(y)

    # Normalise with train statistics
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std  = X_train.std(dim=(0, 1),  keepdim=True)
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    def loader(Xd, yd, shuffle):
        return DataLoader(TensorDataset(Xd, yd), batch_size=BATCH_SIZE, shuffle=shuffle)

    return loader(X_train, y_train, True), loader(X_val, y_val, False), loader(X_test, y_test, False)


# ── Minimal CircadianModel (no mamba / moment dependencies) ───────────────────

def positional_embeddings(shape):
    batch_size, seq_len, embed_dim = shape
    positions = torch.arange(seq_len).unsqueeze(1).float()
    denominators_sin = torch.pow(10000.0, 2 * torch.arange(0, embed_dim, 2) / embed_dim)
    denominators_cos = torch.pow(10000.0, 2 * torch.arange(1, embed_dim, 2) / embed_dim)
    embeddings = torch.zeros(seq_len, embed_dim)
    embeddings[:, 0::2] = torch.sin(positions / denominators_sin)
    embeddings[:, 1::2] = torch.cos(positions / denominators_cos)
    return embeddings.repeat(batch_size, 1, 1)


class CircadianModelDemo(pl.LightningModule):
    """
    Stripped-down CircadianModel for the demo.
    Supports: lstm, transformer, tcn.
    (mamba and moment require optional heavy dependencies.)
    """

    def __init__(self, model_type, params):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type

        params = params.copy()
        self.lr = params.pop("learning_rate")
        hidden  = params.pop("d_model")
        self.params = params

        self.embedder = nn.Linear(INPUT_SIZE, hidden)

        if model_type == "lstm":
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True, **params)
            out = hidden * 2 if params.get("bidirectional") else hidden
            self.classifier = nn.Linear(out, OUTPUT_SIZE)

        elif model_type == "transformer":
            self.encoder     = TransformerEncoder(input_dim=hidden, **params)
            self.classifier  = nn.Linear(hidden, OUTPUT_SIZE)
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden))

        elif model_type == "tcn":
            from pytorch_tcn import TCN
            self.tcn        = TCN(num_inputs=hidden, input_shape="NLC", **params)
            self.classifier = nn.Linear(hidden, OUTPUT_SIZE)

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.embedder(x)

        if self.model_type == "lstm":
            _, (h_n, _) = self.lstm(x)
            h_n = torch.cat((h_n[0], h_n[1]), dim=1) if self.params.get("bidirectional") else h_n[0]
            return self.classifier(h_n)

        elif self.model_type == "transformer":
            x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = x + positional_embeddings(x.shape).to(x.device)
            x = self.encoder(x)
            return self.classifier(x[:, 0, :])

        elif self.model_type == "tcn":
            return self.classifier(self.tcn(x).mean(dim=1))

    def circular_mae(self, y_pred, y_true):
        pred_h = torch.atan2(y_pred[:, 0], y_pred[:, 1]) * 24 / (2 * torch.pi)
        true_h = torch.atan2(y_true[:, 0], y_true[:, 1]) * 24 / (2 * torch.pi)
        diff   = torch.abs(true_h - pred_h)
        return torch.minimum(diff, 24 - diff).mean()

    def _step(self, batch, stage):
        x, y   = batch
        y_hat  = self(x)
        mse    = self.loss_fn(y_hat, y)
        mae    = self.circular_mae(y_hat, y)
        self.log(f"{stage}_loss",      mse, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_hour_loss", mae, on_epoch=True, prog_bar=True)
        return mse

    def training_step(self,   batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self,       batch, _): return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ── Model configs ─────────────────────────────────────────────────────────────

DEMO_PARAMS = {
    "lstm": {
        "d_model":       64,
        "num_layers":    2,
        "bidirectional": True,
        "dropout":       0.1,
        "learning_rate": 1e-3,
    },
    "transformer": {
        "d_model":         64,
        "num_heads":       4,
        "num_layers":      2,
        "dim_feedforward": 128,
        "dropout":         0.1,
        "learning_rate":   1e-3,
    },
    "tcn": {
        "d_model":       64,
        "num_channels":  [64, 64, 64],
        "kernel_size":   4,
        "dropout":       0.1,
        "activation":    "relu",
        "learning_rate": 1e-3,
    },
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic data...")
    X, y = generate_synthetic_data()
    train_loader, val_loader, test_loader = make_splits(X, y)
    print(f"  X shape : {X.shape}  (subjects × timesteps × features)")
    print(f"  y shape : {y.shape}  (subjects × 2 circular targets)\n")

    results = {}
    for model_type, params in DEMO_PARAMS.items():
        print(f"{'─' * 50}")
        print(f"Training: {model_type.upper()}")
        print(f"{'─' * 50}")

        model = CircadianModelDemo(model_type=model_type, params=params)

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")
            ],
        )
        trainer.fit(model, train_loader, val_loader)
        test_results = trainer.test(model, dataloaders=test_loader, verbose=False)

        mae = test_results[0]["test_hour_loss"]
        results[model_type] = mae
        print(f"  → Test circular MAE: {mae:.2f} hours\n")

    print("═" * 50)
    print("Summary — Test circular MAE (hours):")
    for model_type, mae in results.items():
        print(f"  {model_type:<15} {mae:.2f} h")
    print("═" * 50)


if __name__ == "__main__":
    main()
