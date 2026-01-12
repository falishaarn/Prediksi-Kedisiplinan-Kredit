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
with st.sidebar:
    st.title("🛡️ Risk Menu")
    st.markdown("---")
    menu = st.radio("Navigasi", ["🏠 Home", "🔍 Prediksi & Output", "📈 Analytics Dashboard", "🧠 Feature Insights"])
    st.markdown("---")
    st.caption("Versi 2.1 - Fixed Stability")

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
    t1, t2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
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
            os_c, disb_c, saldo_c = get_qcut_label(os_in, df_ref['OS']), get_qcut_label(disb_in, df_ref['Disb']), get_qcut_label(saldo_in, df_ref['Saldo_Rekening'])
            X = pd.DataFrame([[f_enc, os_c, disb_c, saldo_c]], columns=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)'])
            pred = model.predict(X)[0] + 1
            
            st.divider()
            st.subheader(f"Hasil Prediksi Collectibility: {pred}")
            
            # Gauge Indicator
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = pred,
                gauge = {'axis': {'range': [1, 5]},
                         'steps': [{'range': [1, 2], 'color': "green"},
                                   {'range': [2, 5], 'color': "red"}]}))
            st.plotly_chart(fig, use_container_width=True)

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
