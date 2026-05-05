# watermarks/train_cnn.py
import os, math, pickle, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, AUROC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [12031212,1234,5845389,23423,343495,2024,3842834,23402304,482347247,1029237127]

# ---------------- I/O ----------------
def to01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.min() < 0: x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)

def suffix_for(scale: str, position: str, invert: bool) -> str:
    s = ""
    if scale == "neg_one_one": s += "_rescaled"
    if position == "variable": s += "_variablepos"
    if invert: s += "_inverted"
    return s

def dataset_path(artifacts_dir, split_index, base, subset, scale, position, invert) -> str:
    return os.path.join(
        artifacts_dir,
        f"split_{split_index}_{base}_{subset}{suffix_for(scale, position, invert)}.pkl"
    )

def load_split(artifacts_dir, split_index, base, subset, scale, position, invert):
    path = dataset_path(artifacts_dir, split_index, base, subset, scale, position, invert)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # Accept: [data, labels, wm_inds], [data, labels, wm_inds, masks], or ... + filenames
    data, labels = obj[0], obj[1]
    X = to01(data)                         # ensure [0,1]
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    X = np.transpose(X, (0, 3, 1, 2))      # NCHW
    return X, y, path

def make_loaders(artifacts_dir, split_index, base, scale, position, invert, batch_size):
    Xtr, ytr, p_tr = load_split(artifacts_dir, split_index, base, "train", scale, position, invert)
    Xva, yva, p_va = load_split(artifacts_dir, split_index, base, "val",   scale, position, invert)

    tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"train: {p_tr}   N={len(tr)} | val: {p_va}   N={len(va)}")
    return tr_loader, va_loader

# --------------- Model ----------------
class Net(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1),   nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,1,1),  nn.BatchNorm2d(256),nn.ReLU(), nn.MaxPool2d(2),
        )  # 128x128 -> 256x16x16
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096,1028), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1028,num_classes),
        )
    def forward(self, x): return self.head(self.features(x))

# ------------- Train / Eval -----------
def set_seed(seed: int):
    import random, os
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    acc = Accuracy(task="multiclass", num_classes=2).to(DEVICE)
    roc = AUROC(task="multiclass", num_classes=2).to(DEVICE)
    for x, y in loader:
        x = x.to(DEVICE, dtype=torch.float32)
        y = y.to(DEVICE, dtype=torch.long)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0); n += x.size(0)
        acc.update(torch.argmax(probs, 1), y)
        roc.update(probs, y)
    return loss_sum / max(1,n), acc.compute().item(), roc.compute().item()

def train_one_seed(seed, base, tr_loader, va_loader, lr, epochs, wd, momentum, model_dir, tag):
    set_seed(seed)
    model = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    best_val, best_path = math.inf, None

    for ep in range(1, epochs+1):
        t0 = time.time()
        model.train()
        acc = Accuracy(task="multiclass", num_classes=2).to(DEVICE)
        roc = AUROC(task="multiclass", num_classes=2).to(DEVICE)
        train_loss, n = 0.0, 0

        for x, y in tr_loader:
            x = x.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0); n += x.size(0)
            acc.update(torch.argmax(logits, 1), y)
            roc.update(torch.softmax(logits, 1), y)

        v_loss, v_acc, v_roc = evaluate(model, va_loader, criterion)
        tr_loss = train_loss / max(1,n)
        tr_acc = acc.compute().item(); tr_roc = roc.compute().item()

        print(f"[{base}][seed {seed:>2}] ep {ep:03d} "
              f"train: loss {tr_loss:.4f} acc {tr_acc:.3f} auroc {tr_roc:.3f} | "
              f"val: loss {v_loss:.4f} acc {v_acc:.3f} auroc {v_roc:.3f} "
              f"({time.time()-t0:.1f}s)")

        if v_loss < best_val:
            best_val = v_loss
            os.makedirs(model_dir, exist_ok=True)
            best_path = os.path.join(model_dir, f"cnn_{base}_{tag}_seed{seed}.pt")
            torch.save(model.state_dict(), best_path)

    print(f"[{base}] best val loss {best_val:.4f}  saved: {best_path}")

# --------------- Main -----------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-index", type=int, required=True)
    ap.add_argument("--base", choices=["suppressor","confounder","no_watermark","all"], default="all")
    ap.add_argument("--artifacts-dir", type=str, default="./artifacts")
    ap.add_argument("--model-dir", type=str, default="./models")
    ap.add_argument("--position", choices=["fixed","variable"], default="fixed")
    ap.add_argument("--scale", choices=["zero_one","neg_one_one"], default="zero_one")
    ap.add_argument("--invert", action="store_true")

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0,1,2,3,4])
    args, _ = ap.parse_known_args(argv)

    bases = ["suppressor","confounder","no_watermark"] if args.base=="all" else [args.base]
    tag = f"{suffix_for(args.scale, args.position, args.invert)}_split{args.split_index}".lstrip("_")

    for base in bases:
        tr_loader, va_loader = make_loaders(
            args.artifacts_dir, args.split_index, base, args.scale, args.position, args.invert, args.batch_size
        )
        for si in args.seeds:
            train_one_seed(
                SEEDS[si], base, tr_loader, va_loader,
                lr=args.lr, epochs=args.epochs, wd=args.weight_decay, momentum=args.momentum,
                model_dir=args.model_dir, tag=tag
            )

if __name__ == "__main__":
    main()
