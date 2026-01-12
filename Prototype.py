#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

st.set_page_config(page_title="Kredit Predictor Prototype", layout="centered")
st.title("🏦 Sistem Prediksi Collectibility")
st.write("Input data nasabah untuk dianalisis otomatis oleh sistem.")

@st.cache_data
def load_reference_data():
    df = pd.read_csv('Data TA (Kredit).csv')
    return df

@st.cache_resource
def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model('model_xgb.json')
    return model

fcode_list = [
    "CA001", "CCB03", "CS0I1", "KJ001", "KJ002", "KJ003", 
    "KJ004", "KJ006", "KJ007", "KK0A5", "KK0B5", "KP001", 
    "KP003", "KP007", "KP07A", "MG001", "MJ008", "RK007"
]

with st.sidebar:
    st.header("Input Data Mentah")
    fcode_input = st.selectbox("Pilih FCode", fcode_list)
    os_input = st.number_input("Nominal OS", min_value=0.0, value=140562406.0)
    disb_input = st.number_input("Nominal Disbursement", min_value=0.0, value=210000000.0)
    saldo_input = st.number_input("Nominal Saldo Rekening", min_value=0.0, value=2530133.0)

if st.button("Proses & Prediksi"):
    try:
        df_ref = load_reference_data()
        model = load_xgb_model()

        fcode_encoded = fcode_list.index(fcode_input) + 1

        def get_qcut_label(value, series, num_bins=10):
            combined = pd.concat([series, pd.Series([value])], ignore_index=True)
            labels = pd.qcut(combined.rank(method='first'), num_bins, labels=range(1, 11))
            return int(labels.iloc[-1])

        os_cat = get_qcut_label(os_input, df_ref['OS'])
        disb_cat = get_qcut_label(disb_input, df_ref['Disb'])
        saldo_cat = get_qcut_label(saldo_input, df_ref['Saldo_Rekening'])

        input_final = pd.DataFrame({
            'FCode': [fcode_encoded],
            'OS (Category)': [os_cat],
            'Disb (Category)': [disb_cat],
            'Saldo (Category)': [saldo_cat]
        })

        st.subheader("Hasil Transformasi Internal")
        cols = st.columns(4)
        cols[0].metric("FCode", fcode_encoded)
        cols[1].metric("OS Cat", os_cat)
        cols[2].metric("Disb Cat", disb_cat)
        cols[3].metric("Saldo Cat", saldo_cat)

        prediction = model.predict(input_final)
        final_score = prediction[0] + 1

        st.markdown("---")
        st.subheader(f"Hasil Prediksi Collectibility: **{final_score}**")

        if final_score == 1:
            st.success("STATUS: LANCAR")
        elif final_score == 2:
            st.warning("STATUS: DALAM PERHATIAN KHUSUS")
        else:
            st.error("STATUS: MACET")

    except Exception as e:
        st.error(f"Gagal memproses prediksi. Error: {e}")


# In[ ]:




