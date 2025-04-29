import pandas as pd
import numpy as np
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import streamlit as st
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from xgboost import DMatrix, train as xgb_train
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

# AES Setup
key = os.urandom(32)
iv = os.urandom(16)
cipher = Cipher(algorithms.AES(key), modes.CBC(iv))

# --- DATA GENERATION WITH OVERLAP AND NOISE ---
protocols = ['TCP', 'UDP', 'ICMP']
flags = ['SYN', 'ACK', 'FIN', 'RST']
data = []

for _ in range(6000):
    is_anomaly = np.random.rand() < 0.2
    packet_size = np.random.normal(300, 60) if not is_anomaly else np.random.normal(500, 100)
    duration = np.random.normal(0.5, 0.15) if not is_anomaly else np.random.normal(0.8, 0.25)
    
    # 2% label noise
    label = 'anomaly' if is_anomaly else 'normal'
    if np.random.rand() < 0.02:
        label = 'anomaly' if label == 'normal' else 'normal'

    data.append({
        'packet_size': np.clip(packet_size, 60, 1400),
        'duration': np.clip(duration, 0.05, 4.0),
        'protocol': random.choice(protocols),
        'flag': random.choice(flags),
        'src_port': np.random.randint(1024, 60000),
        'dst_port': np.random.randint(1024, 60000),
        'label': label
    })

df = pd.DataFrame(data)
label_encoders = {}
for col in ['protocol', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
df['label'] = df['label'].map({'normal': 0, 'anomaly': 1})

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Less SMOTE: balance but still hard
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Changed from 0.2 to 0.5
X_train, y_train = smote.fit_resample(X_train, y_train)
dtrain = DMatrix(X_train, label=y_train)
dtest = DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.03,
    'max_depth': 2,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'lambda': 2,
    'alpha': 2,
    'seed': 42
}

evals = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb_train(
    params,
    dtrain,
    num_boost_round=60,
    evals=evals,
    early_stopping_rounds=5,
    verbose_eval=False
)

y_pred_prob = bst.predict(dtest)

y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
# Save the trained model
bst.save_model("xgb_anomaly_model.json")

# Optional: Save the label encoders too
joblib.dump(label_encoders, "label_encoders.pkl")

def encrypt_packet(packet_dict):
    serialized = str(packet_dict).encode()
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(serialized) + padder.finalize()
    encryptor = cipher.encryptor()
    return encryptor.update(padded_data) + encryptor.finalize()

def decrypt_packet(encrypted):
    decryptor = cipher.decryptor()
    decrypted_padded = decryptor.update(encrypted) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    decrypted = unpadder.update(decrypted_padded) + unpadder.finalize()
    return eval(decrypted.decode())

incoming_packets = X_test.to_dict(orient='records')
transmitted_packets = []
blocked_packets = []
start_time = time.time()

for packet in incoming_packets:
    packet_dmatrix = DMatrix(pd.DataFrame([packet]))
    prediction = bst.predict(packet_dmatrix)
    if prediction[0] < 0.5:
        encrypted_packet = encrypt_packet(packet)
        transmitted_packets.append(encrypted_packet)
    else:
        blocked_packets.append(packet)

end_time = time.time()
reaction_time = end_time - start_time
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
false_positive_rate = FP / (FP + TN)

# Streamlit UI
st.title("🔐 Secure Network: ML + AES-256")
st.subheader(f"✅ Accuracy: {accuracy*100:.2f}% (Target: 90–99%)")
st.write(classification_report(y_test, y_pred))
st.subheader(f"❗ False Positive Rate: {false_positive_rate*100:.2f}%")
st.subheader(f"⚡ Model Reaction Time: {reaction_time:.2f} seconds")
st.subheader("📦 Encrypted Packets (sample)")
st.write([str(p[:60]) + "..." for p in transmitted_packets[:3]])
st.subheader("🚫 Blocked Packets (sample)")
st.write(blocked_packets[:3])
st.subheader("🔓 Decrypted Example")
st.write(decrypt_packet(transmitted_packets[0]))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.title("Confusion Matrix")
st.pyplot(plt)

# Feature Importance
importance = bst.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'feature': list(importance.keys()),
    'score': list(importance.values())
}).sort_values(by='score', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='score', y='feature')
plt.title("Feature Importance")
st.pyplot(plt)
