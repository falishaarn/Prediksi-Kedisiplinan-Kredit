import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Credit Collectibility Prediction", layout="wide")

# --- LOAD DATA & MODEL ---
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
    st.title("🏦 Risk Analytics & Credit Scoring System")
    st.write("Sistem cerdas analisis kelayakan kredit menggunakan algoritma XGBoost.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 👤 Single Predictor")
        st.write("Analisis mendalam untuk satu nasabah dengan laporan visual.")
        if st.button("Buka Single Predictor", use_container_width=True):
            go_to('single')
            st.rerun()

    with col2:
        st.success("### 📂 Batch Processor")
        st.write("Prediksi massal melalui upload file CSV dan analisis agregat.")
        if st.button("Buka Batch Processor", use_container_width=True):
            go_to('batch')
            st.rerun()

# ==========================================
# HALAMAN 2: SINGLE PREDICTION + ANALYTICS
# ==========================================
elif st.session_state.page == 'single':
    if st.sidebar.button("⬅ Kembali ke Home"):
        go_to('home')
        st.rerun()
        
    st.title("👤 Individual Credit Analysis")
    
    with st.expander("📝 Form Input Data Mentah", expanded=True):
        c1, c2 = st.columns(2)
        fcode_input = c1.selectbox("FCode Nasabah", fcode_list)
        os_input = c1.number_input("Nominal OS", min_value=0.0, value=150000000.0)
        disb_input = c2.number_input("Nominal Disbursement", min_value=0.0, value=200000000.0)
        saldo_input = c2.number_input("Nominal Saldo", min_value=0.0, value=5000000.0)

    if st.button("Jalankan Analisis Cerdas"):
        try:
            df_ref = load_reference_data()
            model = load_xgb_model()

            # Transformasi
            f_enc = fcode_list.index(fcode_input) + 1
            os_cat = get_qcut_label(os_input, df_ref['OS'])
            disb_cat = get_qcut_label(disb_input, df_ref['Disb'])
            saldo_cat = get_qcut_label(saldo_input, df_ref['Saldo_Rekening'])

            input_final = pd.DataFrame([[f_enc, os_cat, disb_cat, saldo_cat]], 
                                      columns=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)'])

            # Prediksi
            prediction = model.predict(input_final)[0] + 1
            
            st.markdown("---")
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric("PREDIKSI KOLEKTIBILITAS", f"Skor: {prediction}")
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    gauge = {'axis': {'range': [1, 5]},
                             'steps': [{'range': [1, 2], 'color': "green"},
                                       {'range': [2, 4], 'color': "orange"},
                                       {'range': [4, 5], 'color': "red"}]}))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_res2:
                st.subheader("💡 Analisis Kontribusi (Feature Importance)")
                # Mengambil importance dari model
                importance = model.feature_importances_
                feat_df = pd.DataFrame({'Fitur': input_final.columns, 'Importance': importance}).sort_values(by='Importance')
                fig_imp = px.bar(feat_df, x='Importance', y='Fitur', orientation='h', title="Faktor Penentu Skor")
                st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# HALAMAN 3: BATCH PREDICTION + DASHBOARD
# ==========================================
elif st.session_state.page == 'batch':
    if st.sidebar.button("⬅ Kembali ke Home"):
        go_to('home')
        st.rerun()

    st.title("📂 Batch Analysis Processor")
    uploaded_file = st.file_uploader("Upload CSV Nasabah", type=["csv"])

    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        if st.button("Proses Seluruh Data"):
            df_ref = load_reference_data()
            model = load_xgb_model()
            
            res = []
            bar = st.progress(0)
            for i, row in df_upload.iterrows():
                f_v = fcode_list.index(row['FCode']) + 1 if row['FCode'] in fcode_list else 1
                o_c = get_qcut_label(row['OS'], df_ref['OS'])
                d_c = get_qcut_label(row['Disb'], df_ref['Disb'])
                s_c = get_qcut_label(row['Saldo_Rekening'], df_ref['Saldo_Rekening'])
                
                pred = model.predict(pd.DataFrame([[f_v, o_c, d_c, s_c]], columns=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)']))[0] + 1
                res.append(pred)
                bar.progress((i + 1) / len(df_upload))
            
            df_upload['Prediksi'] = res
            
            st.divider()
            c_dash1, c_dash2 = st.columns(2)
            with c_dash1:
                st.write("### Distribusi Risiko")
                fig_pie = px.pie(df_upload, names='Prediksi', hole=0.4)
                st.plotly_chart(fig_pie)
            with c_dash2:
                st.write("### Data Hasil Prediksi")
                st.dataframe(df_upload)
                csv = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button("Download Laporan CSV", csv, "report.csv")
