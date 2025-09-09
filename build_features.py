# build_features.py
import pandas as pd

def create_features(vitals_df):
    """Transforms raw time-series vitals into a feature matrix for modeling."""
    df = vitals_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort data to ensure correct temporal calculations
    df = df.sort_values(['patient_id', 'timestamp'])
    
    # Use the last 30 days of data for features
    end_date = df['timestamp'].max()
    start_date = end_date - pd.Timedelta(days=30)
    df_recent = df[df['timestamp'] >= start_date]

    # Aggregate features
    features = df_recent.groupby('patient_id').agg(
        # Mean values
        hr_mean_30d=('heart_rate', 'mean'),
        sbp_mean_30d=('systolic_bp', 'mean'),
        glucose_mean_30d=('blood_glucose', 'mean'),
        
        # Max values (spikes)
        hr_max_30d=('heart_rate', 'max'),
        sbp_max_30d=('systolic_bp', 'max'),
        glucose_max_30d=('blood_glucose', 'max'),
        
        # Standard deviation (variability)
        hr_std_30d=('heart_rate', 'std'),
        sbp_std_30d=('systolic_bp', 'std'),
        
        # Last known value
        last_med_adherence=('med_adherence', 'last')
    ).reset_index()

    features = features.fillna(0) 
    print("Features created successfully.")
    return features

if __name__ == '__main__':
    vitals = pd.read_csv('patient_vitals.csv')
    feature_matrix = create_features(vitals)
    
    outcomes = pd.read_csv('patient_outcomes.csv')
    
    # Merge features and labels
    final_dataset = pd.merge(feature_matrix, outcomes, on='patient_id')
    final_dataset.to_csv('training_dataset.csv', index=False)
    print("Final training dataset saved to training_dataset.csv")
