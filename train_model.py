# train_model.py (multi-model: XGBoost baseline, LSTM advanced, optional TabNet)
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Optional imports for TabNet & PyTorch
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except Exception:
    TABNET_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from models_lstm import VitalsSequenceDataset, LSTMClassifier
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

def train_xgboost(X, y, save_path='risk_model_xgb.pkl'):
    print("Training XGBoost (baseline)...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", use_label_encoder=False, random_state=42, n_jobs=-1)

    param_grid = {
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    grid = GridSearchCV(model, param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # calibrate probabilities
    calibrated = CalibratedClassifierCV(best, method='isotonic', cv=3)
    calibrated.fit(X_train, y_train)

    preds_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (preds_proba > 0.5).astype(int)
    print("XGBoost test AUROC:", roc_auc_score(y_test, preds_proba))
    print("XGBoost test AUPRC:", average_precision_score(y_test, preds_proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # cross-validated AUROC
    cv_auc = cross_val_score(calibrated, X_res, y_res, cv=5, scoring='roc_auc')
    print("Cross-validated AUROC:", cv_auc.mean())

    # save calibrated model
    with open(save_path, 'wb') as f:
        pickle.dump(calibrated, f)
    print("Saved XGBoost calibrated model to", save_path)
    return calibrated

def train_tabnet(X, y, save_path='risk_model_tabnet.pkl'):
    if not TABNET_AVAILABLE:
        print("TabNet not available; skipping TabNet training.")
        return None
    print("Training TabNet (structured features)...")
    # simple train/test split on X,y
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42, stratify=y)
    clf = TabNetClassifier()
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=50, patience=10, batch_size=256, virtual_batch_size=64)
    preds_proba = clf.predict_proba(X_test)[:, 1]
    print("TabNet test AUROC:", roc_auc_score(y_test, preds_proba))
    with open(save_path, 'wb') as f:
        pickle.dump(clf, f)
    print("Saved TabNet model to", save_path)
    return clf

def train_lstm(seq_len=30, save_path='risk_model_lstm.pt', device='cpu', epochs=20):
    """
    Train LSTM on sequences. Save model_state_dict plus metadata (mean/std as plain lists).
    This avoids torch.load unpickling issues for non-primitive objects.
    """
    if not TORCH_AVAILABLE:
        print("PyTorch or models_lstm not available; skipping LSTM training.")
        return None
    print("Preparing sequence dataset for LSTM...")
    ds = VitalsSequenceDataset(vitals_csv='patient_vitals.csv', outcomes_csv='patient_outcomes.csv', seq_len=seq_len)
    # Train/test split by indices
    n = len(ds)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=ds.labels)
    train_ds = torch.utils.data.Subset(ds, train_idx)
    test_ds = torch.utils.data.Subset(ds, test_idx)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = LSTMClassifier(input_size=4, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_auc = 0.0
    for ep in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        # evaluation
        model.eval()
        ys, yps = [], []
        with torch.no_grad():
            for xb, yb, _ in test_loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                ys.extend(yb.numpy().tolist())
                yps.extend(preds.tolist())
        try:
            auc = roc_auc_score(ys, yps)
            ap = average_precision_score(ys, yps)
        except Exception:
            auc = 0.0
            ap = 0.0
        print(f"Epoch {ep+1}/{epochs} loss={np.mean(epoch_losses):.4f} val_AUROC={auc:.4f} val_AUPRC={ap:.4f}")
        if auc > best_auc:
            # save best model weights and normalization (mean/std from ds)
            best_auc = auc
            # convert mean/std (numpy arrays) to plain lists for safe saving
            mean_list = None
            std_list = None
            try:
                if hasattr(ds, 'mean') and ds.mean is not None:
                    mean_list = ds.mean.squeeze().tolist()
                if hasattr(ds, 'std') and ds.std is not None:
                    std_list = ds.std.squeeze().tolist()
            except Exception:
                mean_list = None
                std_list = None

            payload = {
                'model_state_dict': model.state_dict(),
                'mean': mean_list,
                'std': std_list,
                'seq_len': int(ds.seq_len)
            }
            torch.save(payload, save_path)
    print("Saved best LSTM model to", save_path)
    return save_path

def main():
    # Ensure training_dataset.csv exists (structured features)
    if not os.path.exists('training_dataset.csv'):
        print("training_dataset.csv not found. Please run build_features.py first.")
        return

    df = pd.read_csv('training_dataset.csv')
    X = df.drop(columns=['patient_id', 'deteriorated_in_90_days'])
    y = df['deteriorated_in_90_days']

    # Train XGBoost baseline
    xgb_model = train_xgboost(X, y, save_path='risk_model_xgb.pkl')

    # Try TabNet (optional)
    if TABNET_AVAILABLE:
        train_tabnet(X, y, save_path='risk_model_tabnet.pkl')

    # Train LSTM on sequences
    if TORCH_AVAILABLE:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_lstm(seq_len=30, save_path='risk_model_lstm.pt', device=device, epochs=20)
    else:
        print("Skipping LSTM training because PyTorch is not available.")

if __name__ == '__main__':
    main()
