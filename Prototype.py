import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="Credit Collectibility Predictor", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {display: none;}
    
    /* Style Tombol Navigasi Sidebar */
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background-color: transparent;
        text-align: left;
        padding: 12px 20px;
        font-size: 16px;
        color: #31333F;
        transition: 0.3s;
        margin-bottom: 5px;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #e9ecef;
        border: none;
    }
    
    /* Container Metric Card */
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA & MODEL ---
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

# --- SESSION STATE NAVIGASI ---
if 'menu' not in st.session_state:
    st.session_state.menu = "🏠 Home"

def set_menu(name):
    st.session_state.menu = name

# --- SIDEBAR (TANPA BULAT-BULAT) ---
with st.sidebar:
    st.title("Credit Collectibility Predictor")
    st.markdown("---")
    if st.button("🏠 Home"): set_menu("🏠 Home")
    if st.button("🔍 Prediksi & Output"): set_menu("🔍 Prediksi & Output")
    if st.button("📈 Analytics Dashboard"): set_menu("📈 Analytics Dashboard")
    if st.button("🧠 Feature Insights"): set_menu("🧠 Feature Insights")
    st.markdown("---")
    st.caption("v2.5 Enterprise Edition")

df_ref = load_ref()
model = load_xgb_model()
menu = st.session_state.menu

# ==========================================
# LAMAN 1: HOME
# ==========================================
if menu == "🏠 Home":
    st.title("🏦 Credit Collectibility Predictor")
    st.write("Navigasikan sistem menggunakan tombol di sidebar untuk memulai analisis.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Sampel Data", f"{len(df_ref):,}")
    with col_b:
        st.metric("Algoritma AI", "XGBoost Classifier")
    
    st.info("Sistem ini memprediksi status kolektibilitas nasabah (1-5) berdasarkan fitur finansial utama.")

# ==========================================
# LAMAN 2: PREDIKSI & OUTPUT
# ==========================================
elif menu == "🔍 Prediksi & Output":
    st.title("🔍 Collectibility Prediction")
    t1, t2 = st.tabs(["Input Tunggal", "Upload Batch"])
    
    with t1:
        with st.form("form_p"):
            c1, c2 = st.columns(2)
            f_in = c1.selectbox("Pilih FCode", fcode_list)
            os_in = c1.number_input("Nominal OS", value=140562406.0)
            disb_in = c2.number_input("Nominal Disbursement", value=210000000.0)
            saldo_in = c2.number_input("Nominal Saldo", value=2530133.0)
            btn = st.form_submit_button("Analisis Sekarang")
            
        if btn:
            f_enc = fcode_list.index(f_in) + 1
            os_c = get_qcut_label(os_in, df_ref['OS'])
            disb_c = get_qcut_label(disb_in, df_ref['Disb'])
            saldo_c = get_qcut_label(saldo_in, df_ref['Saldo_Rekening'])
            
            X = pd.DataFrame([[f_enc, os_c, disb_c, saldo_c]], columns=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)'])
            pred = model.predict(X)[0] + 1
            
            # --- LOGIKA WARNA PASTEL & STATUS ---
            if pred == 1:
                bg, txt, status = "#D4EDDA", "#155724", "LANCAR"
            elif pred <= 4:
                bg, txt, status = "#FFF3CD", "#856404", "DALAM PENGAWASAN (DPK)"
            else:
                bg, txt, status = "#F8D7DA", "#721C24", "NON-PERFORMING LOAN (NPL)"

            st.markdown(f"""
                <div style="background-color: {bg}; padding: 35px; border-radius: 15px; border: 1px solid {txt}33; text-align: center;">
                    <p style="color: {txt}; font-size: 18px; font-weight: bold; margin: 0; opacity: 0.8;">HASIL PREDIKSI AI</p>
                    <h1 style="color: {txt}; font-size: 64px; margin: 10px 0;">Collectibility {pred}</h1>
                    <p style="color: {txt}; font-size: 26px; font-weight: 500; margin: 0; letter-spacing: 1px;">{status}</p>
                </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.info(f"Analisis dilakukan terhadap nasabah unit {f_in}. Fitur finansial dikonversi ke skala persentil (1-10) sebelum diproses oleh model XGBoost.")

# ==========================================
# LAMAN 3: ANALYTICS
# ==========================================
elif menu == "📈 Analytics Dashboard":
    st.title("📈 Analytics Trends")
    df_line = df_ref.head(50) # Mengambil sampel 50 data agar grafik tetap bersih
    
    st.subheader("Tren Perbandingan OS vs Disbursement")
    fig_line = px.line(df_line, y=['OS', 'Disb'], 
                       title="Trend Line OS & Disbursement (Sampel)",
                       line_shape="spline",
                       color_discrete_map={"OS": "#004aad", "Disb": "#ff4b4b"})
    fig_line.update_layout(hovermode="x unified")
    st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# LAMAN 4: FEATURE INSIGHTS
# ==========================================
elif menu == "🧠 Feature Insights":
    st.title("🧠 Feature Importance Insight")
    
    importances = model.feature_importances_
    features = ['FCode', 'OS (Cat)', 'Disb (Cat)', 'Saldo (Cat)']
    df_imp = pd.DataFrame({'Fitur': features, 'Weight': importances}).sort_values(by='Weight', ascending=True)
    
    col_g, col_t = st.columns([2, 1])
    with col_g:
        fig_imp = px.bar(df_imp, x='Weight', y='Fitur', orientation='h', 
                         title="Bobot Kontribusi Fitur terhadap Prediksi",
                         color_discrete_sequence=['#6c757d'])
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with col_t:
        st.write("### 💡 Insight Strategis")
        top_feature = df_imp.iloc[-1]['Fitur']
        st.info(f"Fitur **{top_feature}** adalah faktor yang paling memengaruhi keputusan model.")
        st.write(f"""
        Dalam model ini, **{top_feature}** memiliki bobot paling tinggi. 
        Ini mengindikasikan bahwa perubahan pada nilai tersebut berkorelasi kuat dengan perubahan tingkat kolektibilitas nasabah.
        """)
