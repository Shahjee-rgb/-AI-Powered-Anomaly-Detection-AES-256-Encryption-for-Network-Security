import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from xgboost import DMatrix
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import os

st.title("📦 Packet Analyzer & Prediction")
st.markdown("Fill in the packet details below to test the model.")

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
loaded_model = xgb.Booster()
loaded_model.load_model("xgb_anomaly_model.json")
loaded_encoders = joblib.load("label_encoders.pkl")

with st.form("test_form"):
    packet_size = st.number_input("Packet Size", min_value=60.0, max_value=1500.0, value=300.0)
    duration = st.number_input("Duration", min_value=0.0, max_value=10.0, value=0.5)
    protocol = st.selectbox("Protocol", loaded_encoders['protocol'].classes_)
    flag = st.selectbox("Flag", loaded_encoders['flag'].classes_)
    src_port = st.number_input("Source Port", min_value=1024, max_value=65535, value=5000)
    dst_port = st.number_input("Destination Port", min_value=1024, max_value=65535, value=8000)
    submitted = st.form_submit_button("Predict")

    if submitted:
        protocol_encoded = loaded_encoders['protocol'].transform([protocol])[0]
        flag_encoded = loaded_encoders['flag'].transform([flag])[0]

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

        pred_prob = loaded_model.predict(input_dmatrix)[0]
        pred_label = "🚨 Anomaly" if pred_prob > 0.5 else "✅ Normal"

        st.success(f"Prediction: {pred_label} (Probability: {pred_prob:.2f})")

        if pred_label == "✅ Normal":
            encrypted = encrypt_packet(input_dict)
            st.info("🔐 Packet encrypted successfully.")
            st.code(str(encrypted[:60]) + "...", language="python")
            decrypted = decrypt_packet(encrypted)
            st.write("🔓 Decrypted Packet:")
            st.json(decrypted)
        else:
            st.warning("🚫 Packet was blocked due to anomaly.")
            
            
