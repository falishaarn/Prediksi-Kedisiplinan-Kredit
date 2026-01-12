import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# --- KONFIGURASI ---
st.set_page_config(page_title="Credit Risk Management System", layout="wide")

@st.cache_data
def load_ref():
    return pd.read_csv('Data TA (Kredit).csv')

@st.cache_resource
def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model('model_xgb.json')
    return model

# DAFTAR FCODE SESUAI URUTAN ENCODE 1-18
fcode_list = [
    "CA001", "CCB03", "CS0I1", "KJ001", "KJ002", "KJ003", 
    "KJ004", "KJ006", "KJ007", "KK0A5", "KK0B5", "KP001", 
    "KP003", "KP007", "KP07A", "MG001", "MJ008", "RK007"
]

def get_qcut_label(value, series):
    combined = pd.concat([series, pd.Series([value])], ignore_index=True)
    labels = pd.qcut(combined.rank(method='first'), 10, labels=range(1, 11))
    return int(labels.iloc[-1])

# --- NAVIGASI SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Risk Nav")
    menu = st.radio("Menu Utama:", 
                    ["🏠 Home", "🔍 Prediksi Collectibility", "📊 Analytics Dashboard", "🧠 Feature Importance"])
    st.markdown("---")
    st.caption("Mode: Enterprise Edition v1.0")

df_ref = load_ref()
model = load_xgb_model()

# ==========================================
# LAMAN 1: HOME
# ==========================================
if menu == "🏠 Home":
    st.title("🏦 Dashboard Risk Management Perbankan")
    st.subheader("Sistem Analisis Kelayakan Kredit & Prediksi Kolektibilitas")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        Platform ini dirancang untuk membantu analis kredit dalam menentukan status kolektibilitas nasabah 
        berdasarkan parameter **FCode, Outstanding, Disbursement, dan Saldo Rekening**.
        
        ### Fitur Utama:
        1. **Prediksi Akurat:** Menggunakan algoritma XGBoost yang dilatih dengan data historis.
        2. **Batch Processing:** Mendukung pengolahan ribuan data nasabah dalam sekali upload.
        3. **Dashboard Analytics:** Visualisasi sebaran data nasabah secara *real-time*.
        4. **Interpretasi Model:** Mengetahui faktor kunci yang mempengaruhi skor nasabah.
        """)
    with col2:
        st.info(f"**Statistik Database:**\n\nTotal Data Referensi: {len(df_ref)} baris")

# ==========================================
# LAMAN 2: PREDIKSI (COLLECTIBILITY 1-5)
# ==========================================
elif menu == "🔍 Prediksi Collectibility":
    st.title("🔍 Prediksi Collectibility Nasabah")
    t1, t2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with t1:
        st.subheader("Input Data Tunggal")
        with st.form("single_form"):
            c1, c2 = st.columns(2)
            f_in = c1.selectbox("Pilih FCode", fcode_list)
            os_in = c1.number_input("Nominal OS", value=150000000.0)
            disb_in = c2.number_input("Nominal Disbursement", value=200000000.0)
            saldo_in = c2.number_input("Nominal Saldo Rekening", value=5000000.0)
            btn = st.form_submit_button("Analisis Sekarang")

        if btn:
            # Transformasi
            f_enc = fcode_list.index(f_in) + 1
            os_cat = get_qcut_label(os_input=os_in, series=df_ref['OS'])
            disb_cat = get_qcut_label(disb_in, df_ref['Disb'])
            saldo_cat = get_qcut_label(saldo_in, df_ref['Saldo_Rekening'])
            
            X = pd.DataFrame([[f_enc, os_cat, disb_cat, saldo_cat]], 
                             columns=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)'])
            
            # Prediksi
            pred = model.predict(X)[0] + 1
            
            # Tampilan Output Utama
            st.divider()
            res_c1, res_c2 = st.columns([1, 2])
            with res_c1:
                st.markdown(f"### Collectibility: **{pred}**")
                if pred == 1: st.success("STATUS: LANCAR")
                elif pred == 2: st.warning("STATUS: DALAM PERHATIAN KHUSUS")
                elif pred == 3: st.warning("STATUS: KURANG LANCAR")
                elif pred == 4: st.warning("STATUS: DIRAGUKAN")
                else: st.error("STATUS: MACET")
            with res_c2:
                # Gauge Chart Visual
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = pred,
                    gauge = {'axis': {'range': [1, 5]},
                             'steps': [{'range': [1, 1.5], 'color': "green"},
                                       {'range': [1.5, 4.5], 'color': "orange"},
                                       {'range': [4.5, 5], 'color': "red"}]}))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("Upload File CSV")
        up_file = st.file_uploader("Pilih file untuk batch prediction", type="csv")
        if up_file:
            df_up = pd.read_csv(up_file)
            if st.button("Proses Batch Prediction"):
                preds = []
                for _, r in df_up.iterrows():
                    f_e = fcode_list.index(r['FCode']) + 1 if r['FCode'] in fcode_list else 1
                    o_c = get_qcut_label(r['OS'], df_ref['OS'])
                    d_c = get_qcut_label(r['Disb'], df_ref['Disb'])
                    s_c = get_qcut_label(r['Saldo_Rekening'], df_ref['Saldo_Rekening'])
                    p = model.predict(pd.DataFrame([[f_e, o_c, d_c, s_c]], columns=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)']))[0] + 1
                    preds.append(p)
                df_up['Collectibility_Result'] = preds
                st.success("Analisis Massal Selesai!")
                st.dataframe(df_up)

# ==========================================
# LAMAN 3: ANALYTICS DASHBOARD
# ==========================================
elif menu == "📊 Analytics Dashboard":
    st.title("📊 Financial & Risk Analytics")
    mode = st.radio("Pilih Level Tampilan:", ["Global Distribution", "Deep Dive Per FCode"])
    
    if mode == "Global Distribution":
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(df_ref, x='OS', title="Distribusi Outstanding (Global)"), use_container_width=True)
        with c2:
            st.plotly_chart(px.box(df_ref, y='Saldo_Rekening', title="Sebaran Saldo Rekening"), use_container_width=True)
    else:
        sel_f = st.selectbox("Pilih FCode yang ingin dianalisis:", fcode_list)
        st.write(f"Menampilkan statistik untuk unit: {sel_f}")
        # Simulasi filter (bisa disesuaikan dengan kolom di dataset kamu)
        st.dataframe(df_ref.sample(5))

# ==========================================
# LAMAN 4: FEATURE IMPORTANCE
# ==========================================
elif menu == "🧠 Feature Importance":
    st.title("🧠 Interpretasi Model AI")
    st.write("Grafik ini menunjukkan fitur mana yang paling menentukan nilai Collectibility nasabah.")
    
    imp = pd.Series(model.feature_importances_, 
                    index=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)']).sort_values()
    
    fig_imp = px.bar(imp, orientation='h', labels={'value':'Importance Score', 'index':'Fitur'})
    st.plotly_chart(fig_imp, use_container_width=True)
    st.info("💡 **Tips:** Jika 'Saldo' memiliki score tertinggi, berarti naik-turunnya saldo adalah faktor utama penentu risiko di model ini.")
