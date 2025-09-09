# generate_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_patients=100):
    """Generates a synthetic dataset for the hackathon."""
    patient_ids = [1000 + i for i in range(num_patients)]
    data = []
    outcomes = []

    for pid in patient_ids:
        # Patient demographics
        is_high_risk = np.random.rand() > 0.7  # 30% of patients are high-risk
        base_hr = 95 if is_high_risk else 75
        base_sbp = 145 if is_high_risk else 125
        base_glucose = 180 if is_high_risk else 100
        adherence = np.random.uniform(0.5, 0.8) if is_high_risk else np.random.uniform(0.8, 1.0)
        
        # Outcome: Did they deteriorate?
        # High-risk patients have a ~60% chance of deteriorating, low-risk ~10%
        deteriorated = 0
        if is_high_risk and np.random.rand() > 0.4:
            deteriorated = 1
        elif not is_high_risk and np.random.rand() > 0.9:
            deteriorated = 1
        
        outcomes.append({'patient_id': pid, 'deteriorated_in_90_days': deteriorated})

        # Generate 180 days of data for each patient
        for day in range(180):
            date = datetime.now() - timedelta(days=180 - day)
            
            # Add noise and trends for high-risk patients
            hr_noise = np.random.normal(0, 5)
            sbp_noise = np.random.normal(0, 8)
            glucose_noise = np.random.normal(0, 15)
            
            hr = base_hr + hr_noise + (day / 30 if is_high_risk else 0)
            sbp = base_sbp + sbp_noise + (day / 40 if is_high_risk else 0)
            glucose = base_glucose + glucose_noise + (day / 20 if is_high_risk else 0)

            data.append({
                'patient_id': pid,
                'timestamp': date,
                'heart_rate': int(hr),
                'systolic_bp': int(sbp),
                'blood_glucose': int(glucose),
                'med_adherence': round(adherence * (1 - (day/1000 if is_high_risk else 0)), 2)
            })

    vitals_df = pd.DataFrame(data)
    outcomes_df = pd.DataFrame(outcomes)
    
    # Save to CSV
    vitals_df.to_csv('patient_vitals.csv', index=False)
    outcomes_df.to_csv('patient_outcomes.csv', index=False)
    print(" Synthetic data generated: patient_vitals.csv, patient_outcomes.csv")

if __name__ == '__main__':
    generate_synthetic_data(num_patients=200)


