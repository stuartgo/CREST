import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = "/data/stuartgo/hunt4/temp_act"
BATCH_SIZE = 64
NUM_WORKERS = 16


def encode_circular(values, period):
    """Encode a value as (sin, cos) components of a circular variable."""
    angle = 2 * torch.pi * values / period
    return torch.sin(angle), torch.cos(angle)


def load_raw(location, model_type, single_day, temperature):
    """Load and preprocess raw tensors for a given configuration."""
    additional_info = torch.load(f"{DATA_DIR}/other_info_{location}.pt")

    if model_type == "moment":
        X = torch.load(f"{DATA_DIR}/embeddings_{location}_{single_day}_{temperature}_moment.pt")
        y = torch.load(f"{DATA_DIR}/y_{location}.pt")
        return X, y, additional_info

    X = torch.load(f"{DATA_DIR}/X_{location}.pt")
    y = torch.load(f"{DATA_DIR}/y_{location}.pt")

    if single_day:
        X = X.unfold(1, 24, 24).flatten(0, 1).permute(0, 2, 1)
        repeats = X.shape[0] // y.shape[0]
        y = y.repeat_interleave(repeats, dim=0)
        additional_info = additional_info.repeat_interleave(repeats, dim=0)

    # Select the activity (index 0) or temperature (index 1) channel
    X = X[:, :, 1:2] if temperature else X[:, :, :1]

    # Append time-of-day sin/cos features
    num_samples, seq_len, _ = X.shape
    start_times = additional_info[:, 2]
    hours = torch.arange(seq_len).unsqueeze(0).expand(num_samples, -1).float()
    hours = hours + start_times.unsqueeze(1).float()
    sin_hour, cos_hour = encode_circular(hours, period=24)
    X = torch.cat([X, sin_hour.unsqueeze(2), cos_hour.unsqueeze(2)], dim=2)

    return X, y, additional_info


def prepare_targets(y):
    """Convert acrophase hours to circular (sin, cos) representation."""
    y = y[:, [1]]  # Select acrophase column
    angle = 2 * torch.pi * y / 24
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    train = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS,persistent_workers=True)
    val   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,persistent_workers=True)
    test  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,persistent_workers=True)
    return train, val, test


def normalise(X_train, X_val, X_test):
    """Normalise all splits using train-set statistics only."""
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std  = X_train.std(dim=(0, 1),  keepdim=True)
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def load_data_kfold(location, model_type, single_day, temperature, train_fraction=1.0):
    """
    Load data and return (X, y, folds), where folds is a list of
    (train_loader, val_loader, test_loader) tuples — one per CV fold.

    train_fraction: proportion of training subjects to use (0, 1].
      Subsampling is done at the subject level to preserve subject-wise integrity.
      Val and test splits are always kept at full size.
    """
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")

    X, y, additional_info = load_raw(location, model_type, single_day, temperature)
    y = prepare_targets(y)
    X, y = X.float(), y.float()

    sids = additional_info[:, 0].numpy()
    unique_sids = np.unique(sids)
    np.random.shuffle(unique_sids)

    kf = KFold(n_splits=5, shuffle=False)  # subjects already shuffled above
    folds = []
    for train_val_idx, test_idx in kf.split(unique_sids):
        train_val_sids = unique_sids[train_val_idx]
        test_sids      = unique_sids[test_idx]

        n_tv = len(train_val_sids)
        train_sids = train_val_sids[:int(0.8 * n_tv)]
        val_sids   = train_val_sids[int(0.8 * n_tv):]

        # Subsample training subjects only
        n_train = max(1, int(len(train_sids) * train_fraction))
        train_sids = set(train_sids[:n_train])
        val_sids   = set(val_sids)

        train_mask = np.isin(sids, list(train_sids))
        val_mask   = np.isin(sids, list(val_sids))
        test_mask  = np.isin(sids, list(test_sids))

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        X_train, X_val, X_test = normalise(X_train, X_val, X_test)
        folds.append(make_loaders(X_train, y_train, X_val, y_val, X_test, y_test))

    return X, y, folds


def load_data_split(location, temperature, single_day):
    """
    Load data and return a single (train_loader, val_loader, test_loader)
    using a 60/20/20 subject-wise split. Used by hyperparameter optimisation.
    """
    # Note: model_type is passed as None here since hyperopt only uses non-moment models
    X, y, additional_info = load_raw(location, model_type=None, single_day=single_day, temperature=temperature)
    y = prepare_targets(y)
    X, y = X.float(), y.float()

    sids = additional_info[:, 0].numpy()
    unique_sids = np.unique(sids)
    np.random.shuffle(unique_sids)

    n = len(unique_sids)
    train_sids = set(unique_sids[:int(0.6 * n)])
    val_sids   = set(unique_sids[int(0.6 * n):int(0.8 * n)])

    train_mask = np.isin(sids, list(train_sids))
    val_mask   = np.isin(sids, list(val_sids))
    test_mask  = ~(train_mask | val_mask)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    X_train, X_val, X_test = normalise(X_train, X_val, X_test)
    return make_loaders(X_train, y_train, X_val, y_val, X_test, y_test)