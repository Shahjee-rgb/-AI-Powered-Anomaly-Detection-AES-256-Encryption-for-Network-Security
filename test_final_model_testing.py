import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from xgboost import DMatrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# AES Setup
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

# Load model and encoders
bst = xgb.Booster()
bst.load_model("xgb_anomaly_model.json")
label_encoders = joblib.load("label_encoders.pkl")

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# Convert y_test to numeric: 'normal' -> 0, 'anomaly' -> 1
y_test = pd.Series(y_test)  # ensure it's a Series for .map()
# Normalize labels: lowercase and strip whitespace
y_test_cleaned = y_test.str.lower().str.strip()

# Optional: print unique labels to debug
print("Unique labels after cleaning:", y_test_cleaned.unique())

# Now apply mapping
label_mapping = {'normal': 0, 'anomaly': 1}
y_test_numeric = y_test_cleaned.map(label_mapping)


# Check for NaN values in y_test_numeric
print(f"Number of NaN values in y_test_numeric before dropna: {y_test_numeric.isna().sum()}")

# Drop NaN values from y_test_numeric
y_test_numeric = y_test_numeric.dropna()

# Ensure no NaN values in y_pred
y_pred_prob = bst.predict(DMatrix(X_test))
y_pred = (y_pred_prob > 0.5).astype(int)
y_pred = y_pred[~np.isnan(y_pred)]  # Ensure no NaN in y_pred

# Debugging: Check the lengths of y_test_numeric and y_pred
print(f"Length of y_test_numeric after dropna: {len(y_test_numeric)}")
print(f"Length of y_pred: {len(y_pred)}")

# Ensure that both have the same length before calculating accuracy
if len(y_test_numeric) == len(y_pred):
    accuracy = accuracy_score(y_test_numeric, y_pred)
else:
    print("Error: Mismatch in length of y_test_numeric and y_pred")
    # Handle the issue accordingly
    accuracy = 0  # Set accuracy to 0 in case of error

# Evaluate Model
report = classification_report(y_test_numeric, y_pred, output_dict=False)
cm = confusion_matrix(y_test_numeric, y_pred)
TN, FP, FN, TP = cm.ravel()
false_positive_rate = FP / (FP + TN)

# Encrypt normal packets, block anomalies
incoming_packets = X_test.to_dict(orient='records')
transmitted_packets, blocked_packets = [], []
start_time = time.time()
for packet in incoming_packets:
    dmat = DMatrix(pd.DataFrame([packet]))
    if bst.predict(dmat)[0] < 0.5:
        transmitted_packets.append(encrypt_packet(packet))
    else:
        blocked_packets.append(packet)
reaction_time = time.time() - start_time

# Streamlit Dashboard UI
st.set_page_config(layout="wide")
st.title("🔐 Secure Network: ML + AES-256")

st.subheader(f"✅ Accuracy: {accuracy*100:.2f}% (Target: 90–99%)")
st.text(report)
st.subheader(f"❗ False Positive Rate: {false_positive_rate*100:.2f}%")
st.subheader(f"⚡ Model Reaction Time: {reaction_time:.2f} seconds")
st.subheader("📦 Encrypted Packets (Sample)")
st.write([str(p[:60]) + "..." for p in transmitted_packets[:3]])
st.subheader("🚫 Blocked Packets (Sample)")
st.write(blocked_packets[:3])
if transmitted_packets:
    st.subheader("🔓 Decrypted Example")
    st.write(decrypt_packet(transmitted_packets[0]))

# Confusion Matrix
st.subheader("📊 Confusion Matrix")
fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1, xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
st.pyplot(fig1)

# Feature Importance
st.subheader("📈 Feature Importance")
importance = bst.get_score(importance_type='weight')
importance_df = pd.DataFrame({'feature': list(importance.keys()), 'score': list(importance.values())}).sort_values(by='score', ascending=False)
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(data=importance_df, x='score', y='feature', ax=ax2)
st.pyplot(fig2)

# Input Form for New Packet
st.markdown("---")
st.header("📥 Test Model with New Packet Input")
with st.form("test_form"):
    packet_size = st.number_input("Packet Size", min_value=60.0, max_value=1500.0, value=300.0)
    duration = st.number_input("Duration", min_value=0.0, max_value=10.0, value=0.5)
    protocol = st.selectbox("Protocol", label_encoders['protocol'].classes_)
    flag = st.selectbox("Flag", label_encoders['flag'].classes_)
    src_port = st.number_input("Source Port", min_value=1024, max_value=65535, value=5000)
    dst_port = st.number_input("Destination Port", min_value=1024, max_value=65535, value=8000)
    submitted = st.form_submit_button("Predict")

    if submitted:
        protocol_encoded = label_encoders['protocol'].transform([protocol])[0]
        flag_encoded = label_encoders['flag'].transform([flag])[0]
        input_dict = {
            'packet_size': packet_size,
            'duration': duration,
            'protocol': protocol_encoded,
            'flag': flag_encoded,
            'src_port': src_port,
            'dst_port': dst_port
        }
        input_df = pd.DataFrame([input_dict])
        input_dmatrix = DMatrix(input_df)
        pred_prob = bst.predict(input_dmatrix)[0]
        pred_label = "🚨 Anomaly" if pred_prob > 0.5 else "✅ Normal"
        st.success(f"Prediction: {pred_label} (Probability: {pred_prob:.2f})")
        if pred_label == "✅ Normal":
            encrypted = encrypt_packet(input_dict)
            st.info("🔐 Packet encrypted successfully.")
            st.code(str(encrypted[:60]) + "...", language="python")
            st.write("🔓 Decrypted Packet:")
            st.json(decrypt_packet(encrypted))
        else:
            st.warning("🚫 Packet was blocked due to anomaly.")
