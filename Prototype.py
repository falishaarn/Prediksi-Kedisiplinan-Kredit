import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Credit Scoring System", layout="centered")

# --- LOAD DATA REFERENSI & MODEL ---
@st.cache_data
def load_reference_data():
    return pd.read_csv('Data TA (Kredit).csv')

@st.cache_resource
def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model('model_xgb.json')
    return model

# DAFTAR FCODE SESUAI URUTAN ENCODE
fcode_list = [
    "CA001", "CCB03", "CS0I1", "KJ001", "KJ002", "KJ003", 
    "KJ004", "KJ006", "KJ007", "KK0A5", "KK0B5", "KP001", 
    "KP003", "KP007", "KP07A", "MG001", "MJ008", "RK007"
]

# --- FUNGSI TRANSFORMASI ---
def get_qcut_label(value, series, num_bins=10):
    combined = pd.concat([series, pd.Series([value])], ignore_index=True)
    labels = pd.qcut(combined.rank(method='first'), num_bins, labels=range(1, 11))
    return int(labels.iloc[-1])

# --- SESSION STATE UNTUK NAVIGASI ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go_to(page_name):
    st.session_state.page = page_name

# ==========================================
# HALAMAN 1: HOME (LANDING PAGE)
# ==========================================
if st.session_state.page == 'home':
    st.title("🏦 Sistem Prediksi Collectibility Kredit")
    st.subheader("Silakan pilih metode input data:")
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### Single Input")
        st.write("Input data nasabah satu per satu secara manual.")
        if st.button("Pilih Single Input", use_container_width=True):
            go_to('single')
            st.rerun()

    with col2:
        st.success("### Batch Input")
        st.write("Upload file CSV untuk prediksi banyak data sekaligus.")
        if st.button("Pilih Batch Input (Upload)", use_container_width=True):
            go_to('batch')
            st.rerun()

# ==========================================
# HALAMAN 2: SINGLE INPUT
# ==========================================
elif st.session_state.page == 'single':
    if st.button("⬅ Kembali ke Home"):
        go_to('home')
        st.rerun()
        
    st.title("👤 Single Data Prediction")
    st.write("---")
    
    with st.form("form_single"):
        fcode_input = st.selectbox("Pilih FCode", fcode_list)
        os_input = st.number_input("Nominal OS", min_value=0.0, value=140562406.0)
        disb_input = st.number_input("Nominal Disbursement", min_value=0.0, value=210000000.0)
        saldo_input = st.number_input("Nominal Saldo Rekening", min_value=0.0, value=2530133.0)
        submit = st.form_submit_button("Proses Prediksi")

    if submit:
        try:
            df_ref = load_reference_data()
            model = load_xgb_model()

            fcode_encoded = fcode_list.index(fcode_input) + 1
            os_cat = get_qcut_label(os_input, df_ref['OS'])
            disb_cat = get_qcut_label(disb_input, df_ref['Disb'])
            saldo_cat = get_qcut_label(saldo_input, df_ref['Saldo_Rekening'])

            input_final = pd.DataFrame({
                'FCode': [fcode_encoded],
                'OS (Category)': [os_cat],
                'Disb (Category)': [disb_cat],
                'Saldo (Category)': [saldo_cat]
            })

            prediction = model.predict(input_final)[0] + 1
            
            st.divider()
            st.subheader(f"Hasil Prediksi: Collectibility {prediction}")
            if prediction == 1: st.success("STATUS: 1 (LANCAR)")
            elif prediction <= 4: st.warning(f"STATUS: {prediction} (DALAM PENGAWASAN)")
            else: st.error("STATUS: 5 (NPL/MACET)")
            
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# HALAMAN 3: BATCH INPUT (UPLOAD)
# ==========================================
elif st.session_state.page == 'batch':
    if st.button("⬅ Kembali ke Home"):
        go_to('home')
        st.rerun()

    st.title("📂 Batch Prediction (Upload File)")
    st.write("Pastikan file CSV memiliki kolom: `FCode`, `OS`, `Disb`, `Saldo_Rekening`")
    
    uploaded_file = st.file_uploader("Unggah File CSV", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write("### Preview Data:")
        st.dataframe(df_upload.head())

        if st.button("Jalankan Prediksi Massal"):
            try:
                df_ref = load_reference_data()
                model = load_xgb_model()
                results = []
                bar = st.progress(0)

                for i, row in df_upload.iterrows():
                    f_val = fcode_list.index(row['FCode']) + 1 if row['FCode'] in fcode_list else 1
                    o_cat = get_qcut_label(row['OS'], df_ref['OS'])
                    d_cat = get_qcut_label(row['Disb'], df_ref['Disb'])
                    s_cat = get_qcut_label(row['Saldo_Rekening'], df_ref['Saldo_Rekening'])
                    
                    input_row = pd.DataFrame([[f_val, o_cat, d_cat, s_cat]], 
                                            columns=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)'])
                    
                    pred = model.predict(input_row)[0] + 1
                    results.append(pred)
                    bar.progress((i + 1) / len(df_upload))

                df_upload['Hasil_Prediksi'] = results
                st.divider()
                st.subheader("Hasil Analisis")
                st.dataframe(df_upload)

                csv = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button("Download Hasil (CSV)", csv, "hasil_prediksi.csv", "text/csv")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file. Error: {e}")
