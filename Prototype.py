import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="Risk Intelligence Analytics", layout="wide")

# --- FIX CSS & AESTHETIC SIDEBAR ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_ref():
    return pd.read_csv('Data TA (Kredit).csv')

@st.cache_resource
def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model('model_xgb.json')
    return model

fcode_list = ["CA001", "CCB03", "CS0I1", "KJ001", "KJ002", "KJ003", "KJ004", "KJ006", "KJ007", "KK0A5", "KK0B5", "KP001", "KP003", "KP007", "KP07A", "MG001", "MJ008", "RK007"]

def get_qcut_label(value, series):
    combined = pd.concat([series, pd.Series([value])], ignore_index=True)
    labels = pd.qcut(combined.rank(method='first'), 10, labels=range(1, 11))
    return int(labels.iloc[-1])

# --- SIDEBAR ---
wwith st.sidebar:
    st.title("🛡️ Risk Menu")
    st.markdown("---")
    if st.button("🏠 Home"): set_menu("🏠 Home")
    if st.button("🔍 Prediksi & Output"): set_menu("🔍 Prediksi & Output")
    if st.button("📈 Analytics Dashboard"): set_menu("📈 Analytics Dashboard")
    if st.button("🧠 Feature Insights"): set_menu("🧠 Feature Insights")
    st.markdown("---")

df_ref = load_ref()
model = load_xgb_model()

# ==========================================
# LAMAN 1: HOME
# ==========================================
if menu == "🏠 Home":
    st.title("🏦 Credit Collectibility Prediction Dashboard")
    st.write("Gunakan menu navigasi untuk melakukan prediksi atau melihat statistik data.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Sampel Data", len(df_ref))
    with col_b:
        st.metric("Model Algoritma", "XGBoost Classifier")

# ==========================================
# LAMAN 2: PREDIKSI & OUTPUT
# ==========================================
elif menu == "🔍 Prediksi & Output":
    st.title("🔍 Prediksi Collectibility")
    t1, t2 = st.tabs(["Input Tunggal", "Upload Batch"])
    
    with t1:
        with st.form("form_p"):
            c1, c2 = st.columns(2)
            f_in = c1.selectbox("Pilih FCode", fcode_list)
            os_in = c1.number_input("Nominal OS", value=140562406.0)
            disb_in = c2.number_input("Nominal Disbursement", value=210000000.0)
            saldo_in = c2.number_input("Nominal Saldo", value=2530133.0)
            btn = st.form_submit_button("Analisis")
            
        if btn:
            f_enc = fcode_list.index(f_in) + 1
            os_c = get_qcut_label(os_in, df_ref['OS'])
            disb_c = get_qcut_label(disb_in, df_ref['Disb'])
            saldo_c = get_qcut_label(saldo_in, df_ref['Saldo_Rekening'])
            
            X = pd.DataFrame([[f_enc, os_c, disb_c, saldo_c]], 
                             columns=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)'])
            
            pred = model.predict(X)[0] + 1
            
            # --- LOGIKA WARNA PASTEL & STATUS ---
            if pred == 1:
                bg_color = "#D4EDDA"  # Hijau Pastel
                text_color = "#155724"
                status = "LANCAR"
            elif pred <= 4:
                bg_color = "#FFF3CD"  # Kuning Pastel
                text_color = "#856404"
                status = "DALAM PENGAWASAN (DPK)"
            else:
                bg_color = "#F8D7DA"  # Merah Pastel
                text_color = "#721C24"
                status = "NON-PERFORMING LOAN (NPL)"

            # --- TAMPILAN KOTAK HASIL ---
            st.divider()
            st.markdown(f"""
                <div style="
                    background-color: {bg_color}; 
                    padding: 30px; 
                    border-radius: 15px; 
                    border: 1px solid {text_color}33;
                    text-align: center;
                    ">
                    <p style="color: {text_color}; font-size: 20px; margin: 0; font-weight: bold;">HASIL ANALISIS AI</p>
                    <h1 style="color: {text_color}; font-size: 60px; margin: 10px 0;">Collectibility {pred}</h1>
                    <p style="color: {text_color}; font-size: 24px; margin: 0; letter-spacing: 2px;">{status}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("") # Spacer
            st.info(f"Nasabah ini dikategorikan ke level {pred} berdasarkan analisis FCode {f_in}, variabel OS, Disbursement, dan Saldo.")
            
# ==========================================
# LAMAN 3: ANALYTICS (LINE CHART)
# ==========================================
elif menu == "📈 Analytics Dashboard":
    st.title("📈 Analytics Trends")
    # Ambil 50 data untuk visualisasi line chart agar tidak terlalu padat
    df_line = df_ref.head(50)
    
    st.subheader("Tren Perbandingan OS vs Disbursement")
    fig_line = px.line(df_line, y=['OS', 'Disb'], title="Trend Line OS & Disbursement")
    st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# LAMAN 4: FEATURE INSIGHTS
# ==========================================
elif menu == "🧠 Feature Insights":
    st.title("🧠 Feature Importance Insight")
    
    # Ambil nilai importance dari model
    importances = model.feature_importances_
    features = ['FCode', 'OS (Cat)', 'Disb (Cat)', 'Saldo (Cat)']
    df_imp = pd.DataFrame({'Fitur': features, 'Weight': importances}).sort_values(by='Weight', ascending=True)
    
    col_g, col_t = st.columns([2, 1])
    with col_g:
        fig_imp = px.bar(df_imp, x='Weight', y='Fitur', orientation='h', title="Bobot Keputusan Model")
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with col_t:
        st.write("### 💡 Insight Strategis")
        # Logika dinamis untuk memberikan insight berdasarkan fitur tertinggi
        top_feature = df_imp.iloc[-1]['Fitur']
        st.info(f"Fitur **{top_feature}** adalah faktor paling dominan dalam menentukan collectibility nasabah.")
        st.write("""
        Artinya, model prediksi sangat memperhatikan perubahan pada nilai ini. 
        Pastikan validasi data pada kolom ini dilakukan dengan ketelitian ekstra.
        """)
