import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Load your original full dataset
df = pd.read_csv("C:\\Users\\syedz\\Desktop\\NETWORKPROJECT\\network_traffic_data.csv")

# Encode categorical columns
label_encoders = {}
for col in ['protocol', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save encoders for Streamlit use
joblib.dump(label_encoders, 'label_encoders.pkl')

# Separate features and label
X = df.drop('label', axis=1)
y = df['label']

# Balance the data with SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Save X_test and y_test for Streamlit
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
