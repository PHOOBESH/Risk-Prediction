# app.py (multi-model aware, full file with SHAP unwrapping)
import streamlit as st
import pandas as pd
import pickle
import shap
import plotly.express as px
import numpy as np
from io import BytesIO
import os

# optional imports
try:
    import torch
    from models_lstm import VitalsSequenceDataset, LSTMClassifier
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except Exception:
    TABNET_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(page_title="AI Risk Prediction Engine", page_icon="🩺", layout="wide")

# --- Utility functions ---
@st.cache_data(ttl=600)
def load_data():
    vitals = pd.read_csv('patient_vitals.csv')
    vitals['timestamp'] = pd.to_datetime(vitals['timestamp'])
    training_data = pd.read_csv('training_dataset.csv')
    return vitals, training_data

@st.cache_resource
def load_xgb_model(path='risk_model_xgb.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_tabnet_model(path='risk_model_tabnet.pkl'):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

# Robust LSTM loader: expects a dict {model_state_dict, mean (list), std (list), seq_len}
@st.cache_resource
def load_lstm_model(path='risk_model_lstm.pt', device='cpu'):
    if not TORCH_AVAILABLE or not os.path.exists(path):
        return None
    import torch
    checkpoint = torch.load(path, map_location=device)
    model = LSTMClassifier(input_size=4, hidden_size=64, num_layers=2, dropout=0.2)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        st.warning(f"Failed to load LSTM state_dict: {e}")
        try:
            model.load_state_dict(checkpoint)
        except Exception:
            return None
    model.eval()
    mean = None
    std = None
    if checkpoint.get('mean') is not None:
        import numpy as _np
        mean = _np.array(checkpoint['mean']).reshape(1, 1, -1)
    if checkpoint.get('std') is not None:
        import numpy as _np
        std = _np.array(checkpoint['std']).reshape(1, 1, -1)
    seq_len = int(checkpoint.get('seq_len', 30))
    return {
        'model': model,
        'mean': mean,
        'std': std,
        'seq_len': seq_len
    }

def predict_proba_for_Xgb(calibrated_model, X_in):
    """
    Robust wrapper: accepts a DataFrame or Series, coerces to numeric, selects model features if possible,
    fills NaNs, and returns positive-class probabilities.
    """
    if isinstance(X_in, pd.Series):
        X_df = X_in.to_frame().T.copy()
    else:
        X_df = X_in.copy()

    for col in ['patient_id', 'risk_score', 'risk_level', 'deteriorated_in_90_days']:
        if col in X_df.columns:
            X_df = X_df.drop(columns=[col])

    feature_order = None
    try:
        # If CalibratedClassifierCV, get underlying estimator's feature names
        # calibrated_model may itself be CalibratedClassifierCV
        inner = None
        if hasattr(calibrated_model, "estimator"):
            inner = calibrated_model.estimator
        elif hasattr(calibrated_model, "base_estimator"):
            inner = calibrated_model.base_estimator
        elif hasattr(calibrated_model, "calibrated_classifiers_") and len(calibrated_model.calibrated_classifiers_)>0:
            # sklearn's calibrated classifiers list contains calibrated clones; try the first
            inner = calibrated_model.calibrated_classifiers_[0]
        if inner is not None and hasattr(inner, "feature_names_in_"):
            feature_order = list(inner.feature_names_in_)
    except Exception:
        feature_order = None

    if feature_order:
        X_df = X_df.reindex(columns=feature_order, fill_value=0)

    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    if feature_order is None:
        X_df = X_df.reindex(sorted(X_df.columns), axis=1)

    try:
        probs = calibrated_model.predict_proba(X_df)[:, 1]
    except Exception:
        preds = calibrated_model.predict(X_df)
        probs = np.array(preds, dtype=float)
    return probs

def predict_proba_for_tabnet(tabnet_model, X_df):
    try:
        probs = tabnet_model.predict_proba(X_df.values)[:,1]
    except Exception:
        probs = np.zeros(len(X_df))
    return probs

def predict_proba_for_lstm(lstm_bundle, patient_ids, vitals_df):
    if lstm_bundle is None:
        return np.zeros(len(patient_ids))
    model = lstm_bundle['model']
    mean = lstm_bundle['mean']
    std = lstm_bundle['std']
    seq_len = lstm_bundle['seq_len']
    all_probs = []
    for pid in patient_ids:
        g = vitals_df[vitals_df['patient_id']==pid].sort_values('timestamp').tail(seq_len)
        if len(g) < seq_len:
            if len(g) > 0:
                pad_count = seq_len - len(g)
                pad_row = g.iloc[[0]].copy()
                pad_df = pd.concat([pad_row]*pad_count, ignore_index=True)
                g = pd.concat([pad_df, g], ignore_index=True)
            else:
                pad_vals = [{'heart_rate':70,'systolic_bp':120,'blood_glucose':100,'med_adherence':1.0}] * seq_len
                g = pd.DataFrame(pad_vals)
        arr = g[['heart_rate','systolic_bp','blood_glucose','med_adherence']].values.astype(float)
        if mean is not None and std is not None:
            try:
                arr = (arr - mean.squeeze()) / std.squeeze()
            except Exception:
                pass
        try:
            import torch
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = model(tensor).numpy().ravel()[0]
            all_probs.append(float(out))
        except Exception:
            all_probs.append(0.0)
    return np.array(all_probs)

def make_patient_report_csv(patient_meta, vitals_df):
    out = BytesIO()
    meta_df = pd.DataFrame([patient_meta])
    meta_df.to_csv(out, index=False)
    out.write(b"\n")
    vitals_df.to_csv(out, index=False)
    out.seek(0)
    return out

# --- SHAP TreeExplainer helper (unwrap calibrated models) ---
def get_tree_explainer_for_model(model, X_sample=None):
    """
    Try to create a shap.TreeExplainer for a (possibly wrapped) tree model.
    Unwrap common sklearn wrappers (CalibratedClassifierCV, pipeline, etc.).
    """
    try:
        # direct attempt
        return shap.TreeExplainer(model, data=(X_sample if X_sample is not None else None))
    except Exception:
        pass

    # Try to unwrap calibrated classifier
    try:
        # CalibratedClassifierCV has attribute 'calibrated_classifiers_' or 'estimator'
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_)>0:
            cal = model.calibrated_classifiers_[0]
            # try to find underlying estimator
            for candidate in ("base_estimator", "estimator", "base_estimator_","estimator_"):
                base = getattr(cal, candidate, None)
                if base is not None:
                    try:
                        return shap.TreeExplainer(base, data=(X_sample if X_sample is not None else None))
                    except Exception:
                        continue
        # top-level wrappers
        for attr in ("base_estimator", "base_estimator_", "estimator", "estimator_", "best_estimator_"):
            base = getattr(model, attr, None)
            if base is not None:
                try:
                    return shap.TreeExplainer(base, data=(X_sample if X_sample is not None else None))
                except Exception:
                    continue
        # try get_booster (xgboost estimator)
        if hasattr(model, "get_booster") and callable(model.get_booster):
            try:
                return shap.TreeExplainer(model.get_booster(), data=(X_sample if X_sample is not None else None))
            except Exception:
                pass
    except Exception:
        pass

    # if we fail, return None to indicate no explainer available
    return None

# --- Load data & models ---
vitals_df, training_df = load_data()
xgb_model = load_xgb_model('risk_model_xgb.pkl')
tabnet_model = load_tabnet_model('risk_model_tabnet.pkl') if TABNET_AVAILABLE else None
lstm_bundle = load_lstm_model('risk_model_lstm.pt') if TORCH_AVAILABLE else None

# Build a small X sample for SHAP background if possible
X_struct_sample = None
try:
    X_struct_sample = training_df.drop(columns=['patient_id','deteriorated_in_90_days']).sample(min(200, len(training_df))).reset_index(drop=True)
except Exception:
    X_struct_sample = None

# Try to build a TreeExplainer for XGBoost (unwrap calibrated wrapper if needed)
explainer = None
if xgb_model is not None:
    explainer = get_tree_explainer_for_model(xgb_model, X_sample=X_struct_sample)
    if explainer is None:
        st.warning("SHAP TreeExplainer could not be created for the current XGBoost model. Fallback will be used for explanations.")

# Sidebar model selection
st.sidebar.title("Model & Settings")
model_opts = ["XGBoost (baseline)"]
if tabnet_model is not None:
    model_opts.append("TabNet (structured)")
if lstm_bundle is not None:
    model_opts.append("LSTM (time-series)")
model_choice = st.sidebar.selectbox("Choose model for predictions", options=model_opts)

risk_threshold = st.sidebar.slider("Risk Score Threshold", 0, 100, 50)
show_low = st.sidebar.checkbox("Show Low-risk patients", value=False)

# Build risk scores using selected model
def annotate_with_model(training_df, model_choice):
    df = training_df.copy()
    if 'deteriorated_in_90_days' not in df.columns:
        df['deteriorated_in_90_days'] = 0
    X_struct = df.drop(columns=['patient_id', 'deteriorated_in_90_days'])
    if model_choice.startswith("XGBoost"):
        if xgb_model is None:
            df['risk_score'] = 0
        else:
            probs = predict_proba_for_Xgb(xgb_model, X_struct)
            df['risk_score'] = (probs*100).round().astype(int)
    elif model_choice.startswith("TabNet"):
        if tabnet_model is None:
            df['risk_score'] = 0
        else:
            probs = predict_proba_for_tabnet(tabnet_model, X_struct)
            df['risk_score'] = (probs*100).round().astype(int)
    elif model_choice.startswith("LSTM"):
        pids = df['patient_id'].tolist()
        probs = predict_proba_for_lstm(lstm_bundle, pids, vitals_df)
        df['risk_score'] = (probs*100).round().astype(int)
    else:
        df['risk_score'] = 0
    df['risk_level'] = pd.cut(df['risk_score'], bins=[-1, 50, 75, 101], labels=['Low', 'Medium', 'High'])
    return df

training_df = annotate_with_model(training_df, model_choice)

# Rest of the UI
st.title("🩺 AI-Driven Risk Prediction for Chronic Care")
st.markdown(f"Model in use: **{model_choice}** — Use the What-If sliders to simulate interventions live.")

# Cohort pie chart and table
st.header("Cohort Risk Overview")
risk_counts = training_df['risk_level'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0)
fig_pie = px.pie(names=risk_counts.index, values=risk_counts.values, title="Risk-level Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

df_display = training_df[['patient_id', 'risk_score', 'risk_level']]
if not show_low:
    filtered_df = df_display[df_display['risk_level'] != 'Low']
else:
    filtered_df = df_display[df_display['risk_score'] >= risk_threshold]
st.dataframe(filtered_df.sort_values('risk_score', ascending=False), use_container_width=True)

# Patient Detail View
st.header("🔬 Patient Deep Dive")
selected_patient_id = st.selectbox("Select a Patient ID", training_df['patient_id'].unique())

if selected_patient_id:
    patient_idx = training_df[training_df['patient_id'] == selected_patient_id].index[0]
    patient_vitals = vitals_df[vitals_df['patient_id'] == selected_patient_id].sort_values('timestamp')
    patient_row = training_df.loc[patient_idx]
    patient_risk_score = int(patient_row['risk_score'])
    patient_risk_level = str(patient_row['risk_level'])

    col1, col2 = st.columns([2,3])
    with col1:
        st.subheader(f"Patient ID: {selected_patient_id}")
        st.metric(label="Predicted Risk of Deterioration (90d)", value=f"{patient_risk_score}%", delta=patient_risk_level)
        st.info("**Recommended Action:** " +
                ("Immediate review & possible hospitalization" if patient_risk_level == 'High' else
                 "Telehealth follow-up recommended" if patient_risk_level == 'Medium' else
                 "Routine monitoring"))
        csv_bytes = make_patient_report_csv(patient_row.to_dict(), patient_vitals)
        st.download_button("Download patient report (CSV)", data=csv_bytes, file_name=f"patient_{selected_patient_id}_report.csv", mime="text/csv")

        st.markdown("### 🔁 Simulate Intervention (What-if)")
        X_cols = training_df.drop(columns=['patient_id','deteriorated_in_90_days']).columns.tolist() if 'deteriorated_in_90_days' in training_df.columns else []
        if len(X_cols)==0:
            st.write("No structured feature matrix available for interventions (LSTM model uses raw sequences).")
        else:
            interventions = {}
            cols_for_intervention = [c for c in X_cols if 'adherence' in c or 'glucose' in c or 'hr' in c or 'sbp' in c]
            if not cols_for_intervention:
                st.write("No adjustable features available in feature matrix.")
            else:
                for feat in cols_for_intervention:
                    cur = float(patient_row.get(feat, 0.0))
                    if 'adherence' in feat:
                        new = st.slider(feat, 0.0, 1.0, float(cur), step=0.01)
                    elif 'glucose' in feat or 'gluc' in feat:
                        new = st.slider(feat, 50, 400, int(max(50, cur)), step=1)
                    elif 'hr' in feat:
                        new = st.slider(feat, 40, 160, int(max(40, cur)), step=1)
                    elif 'sbp' in feat:
                        new = st.slider(feat, 80, 240, int(max(80, cur)), step=1)
                    else:
                        new = st.slider(feat, cur * 0.5, cur * 1.5, cur)
                    interventions[feat] = new

                if model_choice.startswith("XGBoost"):
                    X_struct_series = patient_row.drop(labels=['patient_id','deteriorated_in_90_days']).copy()
                    for k, v in interventions.items():
                        if k in X_struct_series.index:
                            X_struct_series.loc[k] = v
                    X_struct_df = X_struct_series.to_frame().T
                    proba_orig = predict_proba_for_Xgb(xgb_model, X_struct_df)[0] if xgb_model is not None else 0.0
                    proba_new = predict_proba_for_Xgb(xgb_model, X_struct_df)[0] if xgb_model is not None else 0.0
                    st.write(f"Original risk: **{proba_orig*100:.1f}%** → After intervention: **{proba_new*100:.1f}%**")
                else:
                    st.write("Intervention simulation for selected model is only supported for structured models (XGBoost/TabNet).")

        with col2:
            st.subheader("Key Risk Drivers (Local explanation)")
            # Only attempt SHAP for XGBoost (tree models) and if we have an explainer
            if model_choice.startswith("XGBoost") and xgb_model is not None and explainer is not None:
                try:
                    # Build X_row from structured features and select only numeric model features
                    X_struct = training_df.drop(columns=['patient_id', 'deteriorated_in_90_days'])
                    X_row = X_struct.loc[[patient_idx]]

                    # Try to determine feature order used by the trained estimator (robust to wrappers)
                    feature_order = None
                    try:
                        inner = None
                        if hasattr(xgb_model, "estimator"):
                            inner = xgb_model.estimator
                        elif hasattr(xgb_model, "base_estimator"):
                            inner = xgb_model.base_estimator
                        elif hasattr(xgb_model, "calibrated_classifiers_") and len(xgb_model.calibrated_classifiers_) > 0:
                            inner = xgb_model.calibrated_classifiers_[0]
                        if inner is not None and hasattr(inner, "feature_names_in_"):
                            feature_order = list(inner.feature_names_in_)
                    except Exception:
                        feature_order = None

                    # Reindex to the model's expected features (fill missing with 0)
                    if feature_order:
                        X_row_clean = X_row.reindex(columns=feature_order, fill_value=0)
                    else:
                        # Fallback: drop non-numeric columns then coerce remaining to numeric
                        X_row_clean = X_row.select_dtypes(include=[np.number])
                        if X_row_clean.shape[1] == 0:
                            # If selective dtype failed, coerce everything to numeric (objects -> NaN -> fill 0)
                            X_row_clean = X_row.apply(pd.to_numeric, errors='coerce').fillna(0)
                        else:
                            X_row_clean = X_row_clean.apply(pd.to_numeric, errors='coerce').fillna(0)

                    # Ensure dataframe rows/columns are numeric and not empty
                    if X_row_clean.shape[1] == 0:
                        st.write("No numeric features available for SHAP explanation.")
                    else:
                        # Compute SHAP values on the cleaned DataFrame
                        shap_vals = explainer.shap_values(X_row_clean)
                        # shap may return a list (per-class) or array
                        if isinstance(shap_vals, list):
                            sv = shap_vals[0]
                        else:
                            sv = shap_vals
                        # sv shape usually (n_samples, n_features)
                        arr = sv[0] if getattr(sv, "ndim", 1) == 2 else sv
                        shap_df = pd.DataFrame({'Feature': X_row_clean.columns, 'SHAP_Value': arr})
                        shap_df = shap_df.sort_values('SHAP_Value', key=abs, ascending=False)
                        st.bar_chart(shap_df.set_index('Feature')['SHAP_Value'])
                except Exception as e:
                    # SHAP failed — fallback to feature_importances_ when available
                    st.write("SHAP explanation failed:", e)
                    try:
                        # Try to locate underlying estimator with feature_importances_
                        base = getattr(xgb_model, "estimator", None) or getattr(xgb_model, "base_estimator", None) or xgb_model
                        feat_imp = getattr(base, "feature_importances_", None)
                        if feat_imp is not None:
                            X_struct = training_df.drop(columns=['patient_id', 'deteriorated_in_90_days'])
                            fi_df = pd.DataFrame({'Feature': X_struct.columns, 'Importance': feat_imp}).sort_values('Importance', ascending=False)
                            st.bar_chart(fi_df.set_index('Feature')['Importance'])
                        else:
                            st.write("No feature importances available for fallback.")
                    except Exception as e2:
                        st.write("Fallback also failed:", e2)
            else:
                st.write("Feature-level explanations (SHAP) currently shown for XGBoost only. For LSTM/TabNet, consider adding model-specific attribution methods (Integrated Gradients / attention weights).")

    st.subheader("Vital Signs Trend (last 180 days)")
    fig_vitals = px.line(patient_vitals, x='timestamp', y=['heart_rate','systolic_bp','blood_glucose'],
                         title="Patient Vitals Over Last 180 Days", labels={'value':'Reading','variable':'Vital Sign'})
    st.plotly_chart(fig_vitals, use_container_width=True)

# model evaluation expander
with st.expander("Model evaluation & calibration"):
    st.write("Displayed metrics correspond to the model selected (if precomputed during training).")
    try:
        y_true = training_df['deteriorated_in_90_days']
        y_prob = training_df['risk_score'] / 100.0
        from sklearn.metrics import roc_auc_score, average_precision_score
        st.write(f"Approx AUROC: **{roc_auc_score(y_true, y_prob):.3f}**")
        st.write(f"Approx AUPRC: **{average_precision_score(y_true, y_prob):.3f}**")
    except Exception as e:
        st.write("Failed to compute evaluation metrics:", e)
