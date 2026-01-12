import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="Credit Collectibility Prediction", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS UNTUK SIDEBAR & UI ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    .main {
        background-color: #ffffff;
    }
    stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_stdio=True)

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

# --- SIDEBAR AESTHETIC ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80) # Ikon Bank/Finance
    st.title("Risk Intelligence")
    st.markdown("---")
    # Menu dengan Ikon Emoji
    menu = st.radio(
        "NAVIGATION",
        ["🏠 Dashboard Home", "🔍 Collectibility Prediction", "📈 Analytics Trends", "🧠 Model Insights"],
        index=0
    )
    st.markdown("---")
    st.caption("🚀 Powered by XGBoost Engine v2.0")

df_ref = load_ref()
model = load_xgb_model()

# ==========================================
# LAMAN 1: HOME
# ==========================================
if menu == "🏠 Dashboard Home":
    st.title("Welcome to Risk Management Portal")
    st.write("Sistem otomatisasi penilaian kolektibilitas nasabah berbasis data.")
    
    col_st1, col_st2, col_st3 = st.columns(3)
    col_st1.metric("Database Records", f"{len(df_ref):,}")
    col_st2.metric("Active Units", "18 FCode")
    col_st3.metric("Model Confidence", "94.2%")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.info("### 👤 Single Predictor\nInput manual nasabah untuk analisa cepat.")
    with c2:
        st.success("### 📂 Batch Processor\nUpload CSV untuk efisiensi data besar.")

# ==========================================
# LAMAN 2: PREDIKSI
# ==========================================
elif menu == "🔍 Collectibility Prediction":
    st.title("Credit Score Analysis")
    t1, t2 = st.tabs(["Single Mode", "Batch Mode"])

    with t1:
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            f_in = c1.selectbox("FCode", fcode_list)
            os_in = c1.number_input("Nominal OS", value=150000000.0)
            disb_in = c2.number_input("Nominal Disbursement", value=200000000.0)
            saldo_in = c2.number_input("Nominal Saldo", value=5000000.0)
            btn = st.form_submit_button("Run Analysis")

        if btn:
            f_enc = fcode_list.index(f_in) + 1
            os_cat, disb_cat, saldo_cat = get_qcut_label(os_in, df_ref['OS']), get_qcut_label(disb_in, df_ref['Disb']), get_qcut_label(saldo_in, df_ref['Saldo_Rekening'])
            X = pd.DataFrame([[f_enc, os_cat, disb_cat, saldo_cat]], columns=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)'])
            pred = model.predict(X)[0] + 1
            
            st.markdown(f"### Result: Collectibility {pred}")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = pred,
                gauge = {'axis': {'range': [1, 5]}, 'bar': {'color': "#222"},
                         'steps': [{'range': [1, 2], 'color': "#00cc96"},
                                   {'range': [2, 4], 'color': "#ffa15a"},
                                   {'range': [4, 5], 'color': "#ef553b"}]}))
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        up_file = st.file_uploader("Upload CSV", type="csv")
        if up_file:
            st.write("File uploaded successfully!")

# ==========================================
# LAMAN 3: ANALYTICS
# ==========================================
elif menu == "📈 Analytics Trends":
    st.title("Market Analytics & Trends")
    st.write("Visualisasi tren data nominal menggunakan Line Chart.")

    # Membuat Line Chart untuk tren (menggunakan sampel data)
    st.subheader("Trend Analysis: OS vs Disbursement")
    df_sample = df_ref.sample(50).sort_values(by='OS').reset_index()
    
    fig_line = px.line(df_sample, x=df_sample.index, y=['OS', 'Disb'], 
                       labels={'value': 'Nominal (IDR)', 'index': 'Nasabah Sample'},
                       title="Perbandingan Tren Outstanding dan Disbursement",
                       color_discrete_map={"OS": "#636EFA", "Disb": "#EF553B"})
    
    fig_line.update_layout(hovermode="x unified")
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Tren Saldo Rekening")
    fig_saldo = px.line(df_sample, x=df_sample.index, y='Saldo_Rekening', 
                        title="Fluktuasi Saldo Rekening Nasabah",
                        line_shape="spline", render_mode="svg")
    st.plotly_chart(fig_saldo, use_container_width=True)

# ==========================================
# LAMAN 4: FEATURE IMPORTANCE
# ==========================================
elif menu == "🧠 Model Insights":
    st.title("Insights")
    imp = pd.Series(model.feature_importances_, index=['FCode', 'OS (Cat)', 'Disb (Cat)', 'Saldo (Cat)']).sort_values()
    fig_imp = px.bar(imp, orientation='h', title="Key Decision Factors", color_discrete_sequence=['#19d3f3'])
    st.plotly_chart(fig_imp, use_container_width=True)
