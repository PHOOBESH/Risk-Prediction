# models_lstm.py
# Helper PyTorch LSTM model and dataset utilities for sequence modelling.

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class VitalsSequenceDataset(Dataset):
    """
    Builds sequences per patient from patient_vitals.csv.
    Each sample: last_seq_len timesteps of (heart_rate, systolic_bp, blood_glucose, med_adherence)
    Label: deteriorated_in_90_days from patient_outcomes.csv
    """
    def __init__(self, vitals_csv='patient_vitals.csv', outcomes_csv='patient_outcomes.csv', seq_len=30, transform=None):
        vit = pd.read_csv(vitals_csv)
        vit['timestamp'] = pd.to_datetime(vit['timestamp'])
        self.seq_len = seq_len
        self.transform = transform

        # keep relevant cols and sort
        vit = vit.sort_values(['patient_id', 'timestamp'])
        # group and build last seq_len records per patient
        groups = vit.groupby('patient_id')
        data = []
        ids = []
        for pid, g in groups:
            # take last seq_len rows
            g_last = g.tail(seq_len)
            # if less than seq_len, pad at beginning with last value
            if len(g_last) < seq_len:
                # repeat first row to pad
                pad_count = seq_len - len(g_last)
                pad_row = g_last.iloc[[0]].copy()
                pad_df = pd.concat([pad_row] * pad_count, ignore_index=True)
                g_last = pd.concat([pad_df, g_last], ignore_index=True)
            # select features
            arr = g_last[['heart_rate', 'systolic_bp', 'blood_glucose', 'med_adherence']].values.astype(float)
            data.append(arr)
            ids.append(pid)

        self.sequences = np.stack(data)  # shape (n_patients, seq_len, n_features)
        self.patient_ids = np.array(ids)

        # load outcomes as labels aligned by patient_id
        out = pd.read_csv(outcomes_csv)
        out_map = dict(zip(out['patient_id'], out['deteriorated_in_90_days']))
        self.labels = np.array([out_map.get(pid, 0) for pid in self.patient_ids]).astype(np.float32)

        # normalize features per feature across dataset (simple z-score)
        self.mean = self.sequences.mean(axis=(0,1), keepdims=True)
        self.std = self.sequences.std(axis=(0,1), keepdims=True) + 1e-8
        self.sequences = (self.sequences - self.mean) / self.std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)  # shape (seq_len, n_features)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        pid = int(self.patient_ids[idx])
        if self.transform:
            x = self.transform(x)
        return x, y, pid

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)  # out: (batch, seq_len, hidden)
        # use last hidden state
        last = out[:, -1, :]  # (batch, hidden_size)
        x = self.dropout(last)
        x = self.fc(x).squeeze(-1)
        return x  # (batch,)
