import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from xgboost import DMatrix, train as xgb_train
import os
import time
import joblib

# AES setup
key = os.urandom(32)
iv = os.urandom(16)
cipher = Cipher(algorithms.AES(key), modes.CBC(iv))

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

# Generate synthetic packet data
protocols = ['TCP', 'UDP', 'ICMP']
flags = ['SYN', 'ACK', 'FIN', 'RST']
data = []

for _ in range(6000):
    is_anomaly = np.random.rand() < 0.2
    packet_size = np.random.normal(300, 40) if not is_anomaly else np.random.normal(600, 60)
    duration = np.random.normal(0.4, 0.1) if not is_anomaly else np.random.normal(1.2, 0.3)

    label = 'anomaly' if is_anomaly else 'normal'
    if np.random.rand() < 0.01:
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

# Label encoding
label_encoders = {}
for col in ['protocol', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
df['label'] = df['label'].map({'normal': 0, 'anomaly': 1})

X = df.drop('label', axis=1)
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Oversample to balance
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# DMatrix for XGBoost
dtrain = DMatrix(X_train, label=y_train)
dtest = DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'lambda': 1,
    'alpha': 1,
    'seed': 42
}

evals = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb_train(
    params,
    dtrain,
    num_boost_round=200,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=False
)

# Save model and encoders
bst.save_model("xgb_anomaly_model.json")
joblib.dump(label_encoders, "label_encoders.pkl")
X_test.to_csv("X_test.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

# Predict and evaluate
y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# AES Packet handling
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
TN, FP, FN, TP = cm.ravel()
false_positive_rate = FP / (FP + TN)
print(f"🔁 Reaction time: {reaction_time:.2f}s")
print(f"❗ False Positive Rate: {false_positive_rate*100:.2f}%")
