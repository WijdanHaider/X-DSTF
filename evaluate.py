import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from tqdm import tqdm


def evaluate(model, dataloader, device):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for spatial, freq, labels in tqdm(dataloader):
            spatial = spatial.to(device)
            freq = freq.to(device)
            labels = labels.to(device)

            logits = model(spatial, freq)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs > 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)

    return {
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
