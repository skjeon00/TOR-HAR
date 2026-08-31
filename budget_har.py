import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Reproducibility / normalization
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def standardize_from_train(X_train, X_val, X_test):
    """Channel-wise z-score using training data only."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6

    def z(x):
        return ((x - mean) / std).astype(np.float32)

    return z(X_train), z(X_val), z(X_test)


# =====================================================================
# Part A. Temporal-resolution backbone: Light1DCNN
# =====================================================================

class HARDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32)).float()
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64)).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Light1DCNN(nn.Module):
    """
    Light 1D CNN used with the same topology across temporal resolutions.

    AdaptiveAvgPool1d makes the model compatible with different input lengths
    (e.g., full, 1/2, 1/4, and 1/8 temporal resolutions).
    """

    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(96, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(96, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


@torch.no_grad()
def evaluate_light(model, loader, n_classes: int):
    model.eval()

    ys, preds = [], []

    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)

        logits = model(X)
        pred = logits.argmax(dim=1).cpu().numpy()

        ys.append(y.numpy())
        preds.append(pred)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "class_f1": f1_score(
            y_true,
            y_pred,
            labels=np.arange(n_classes),
            average=None,
            zero_division=0,
        ),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_light_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    *,
    in_ch: int,
    n_classes: int,
    seed: int = 42,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 25,
    patience: int = 6,
):
    """
    Training/test loop matching the temporal-resolution experiments.

    Inputs are expected in [N, C, T] format and should already represent the
    desired temporal resolution.
    """
    seed_everything(seed)

    train_ds = HARDataset(X_train, y_train)
    val_ds = HARDataset(X_val, y_val)
    test_ds = HARDataset(X_test, y_test)

    common = dict(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = Light1DCNN(in_ch=in_ch, n_classes=n_classes).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()

        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_light(model, val_loader, n_classes)
        val_f1 = val_metrics["macro_f1"]
        scheduler.step(val_f1)

        if val_f1 > best_val + 1e-4:
            best_val = val_f1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_light(model, test_loader, n_classes)

    return model, {
        "val_macro_f1": best_val,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "class_f1": test_metrics["class_f1"],
    }


# =====================================================================
# Part B. Budget-matched backbone: Patch CNN + BiGRU
# =====================================================================

def patchify(X, patch_len: int):
    """
    Convert [N, C, T] -> [N, M, C, patch_len].
    The source experiments use M=8 patches.
    """
    X = np.asarray(X, dtype=np.float32)
    N, C, T = X.shape

    if T % patch_len != 0:
        raise ValueError("Input length must be divisible by patch_len.")

    n_patches = T // patch_len
    return X.reshape(N, C, n_patches, patch_len).transpose(0, 2, 1, 3)


def energy_scores(patches):
    # patches: [N, M, C, L]
    d = np.diff(patches, axis=-1)
    motion = np.mean(np.abs(d), axis=(2, 3))
    variability = np.mean(np.var(patches, axis=-1), axis=2)
    return (motion + 0.10 * variability).astype(np.float32)


def uniform_indices(n_patches: int, k: int):
    if k == n_patches:
        return np.arange(n_patches, dtype=np.int64)
    if k == 1:
        return np.array([n_patches // 2], dtype=np.int64)

    idx = np.round(np.linspace(0, n_patches - 1, k)).astype(np.int64)
    idx = np.unique(idx)

    if len(idx) < k:
        extra = [i for i in range(n_patches) if i not in idx]
        idx = np.concatenate([idx, np.asarray(extra[: k - len(idx)])])

    return np.sort(idx[:k])


def contiguous_indices(n_patches: int, k: int):
    start = (n_patches - k) // 2
    return np.arange(start, start + k, dtype=np.int64)


def random_indices(n_samples: int, n_patches: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, k), dtype=np.int64)

    for i in range(n_samples):
        out[i] = np.sort(
            rng.choice(n_patches, size=k, replace=False)
        )

    return out


def energy_indices(X, patch_len: int, k: int):
    p = patchify(X, patch_len)
    n_patches = p.shape[1]
    s = energy_scores(p)
    idx = np.argpartition(s, kth=n_patches - k, axis=1)[:, -k:]
    return np.sort(idx, axis=1).astype(np.int64)


def make_patches_and_positions(
    X,
    *,
    method: str,
    k: int,
    patch_len: int,
    seed: int,
):
    """
    Return:
        patches   [N, K, C, patch_len]
        positions [N, K]

    Supported methods from the source notebook:
        full, downsample, contiguous, uniform, random, energy

    At a fixed K, all methods pass exactly K patches through the patch encoder.
    """
    X = np.asarray(X, dtype=np.float32)
    N, C, T = X.shape

    if T % patch_len != 0:
        raise ValueError("Input length must be divisible by patch_len.")

    n_patches = T // patch_len

    if not 1 <= k <= n_patches:
        raise ValueError("k must be between 1 and n_patches.")

    if method == "full":
        if k != n_patches:
            raise ValueError("full requires k == n_patches.")
        p = patchify(X, patch_len)
        pos = np.tile(np.arange(n_patches, dtype=np.int64), (N, 1))
        return p.astype(np.float32), pos

    if method == "downsample":
        factor = n_patches // k
        if n_patches % k != 0:
            raise ValueError("downsample requires n_patches divisible by k.")

        x = X[:, :, ::factor]
        x = x[:, :, : k * patch_len]

        p = x.reshape(N, C, k, patch_len).transpose(0, 2, 1, 3)

        if k == 1:
            base = np.array([n_patches // 2], dtype=np.int64)
        else:
            base = np.round(
                np.linspace(0, n_patches - 1, k)
            ).astype(np.int64)

        pos = np.tile(base, (N, 1))
        return p.astype(np.float32), pos

    p_all = patchify(X, patch_len)

    if method == "uniform":
        idx = uniform_indices(n_patches, k)
        return (
            p_all[:, idx].astype(np.float32),
            np.tile(idx, (N, 1)),
        )

    if method == "contiguous":
        idx = contiguous_indices(n_patches, k)
        return (
            p_all[:, idx].astype(np.float32),
            np.tile(idx, (N, 1)),
        )

    if method == "random":
        idx = random_indices(N, n_patches, k, seed)
    elif method == "energy":
        idx = energy_indices(X, patch_len, k)
    else:
        raise ValueError(f"Unknown method: {method}")

    p = np.empty((N, k, C, patch_len), dtype=np.float32)
    for i in range(N):
        p[i] = p_all[i, idx[i]]

    return p, idx


class BudgetDataset(Dataset):
    def __init__(self, X, y, *, method, k, patch_len, seed):
        patches, positions = make_patches_and_positions(
            X,
            method=method,
            k=k,
            patch_len=patch_len,
            seed=seed,
        )

        self.patches = torch.from_numpy(patches).float()
        self.positions = torch.from_numpy(positions).long()
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64)).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.patches[idx], self.positions[idx], self.y[idx]


class PatchEncoder(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int = 96):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 48, 5, padding=2, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),

            nn.Conv1d(48, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, emb_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.pool(self.net(x)).squeeze(-1)


class BudgetHAR(nn.Module):
    """
    Patch CNN + positional embedding + bidirectional GRU.

    Source defaults:
        emb_dim=96
        pos_dim=16
        hidden=64
        n_patches=8
    """

    def __init__(
        self,
        *,
        in_channels: int,
        n_classes: int,
        n_patches: int = 8,
        emb_dim: int = 96,
        pos_dim: int = 16,
        hidden: int = 64,
    ):
        super().__init__()

        self.patch_encoder = PatchEncoder(in_channels, emb_dim)
        self.pos_emb = nn.Embedding(n_patches, pos_dim)

        self.gru = nn.GRU(
            input_size=emb_dim + pos_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, n_classes),
        )

    def forward(self, patches, positions):
        B, K, C, L = patches.shape

        z = patches.reshape(B * K, C, L)
        z = self.patch_encoder(z)
        z = z.reshape(B, K, -1)

        z = torch.cat([z, self.pos_emb(positions)], dim=-1)
        z, _ = self.gru(z)
        z = z.mean(dim=1)

        return self.head(z)
