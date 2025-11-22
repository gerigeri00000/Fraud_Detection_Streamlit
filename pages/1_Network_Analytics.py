import streamlit as st
import pandas as pd
from pyvis.network import Network
import networkx as nx
from network_analysis import build_claim_graph, calculate_graph_risk
import requests
import json

if "inference_done" not in st.session_state:
    st.session_state.inference_done = False

if "inference_results" not in st.session_state:
    st.session_state.inference_results = None

if "df_uploaded" not in st.session_state:
    st.session_state.df_uploaded = None

API_BASE = st.secrets["API_BASE"] #st.secrets["API_BASE"]"http://localhost:8989"
st.set_page_config(layout="wide", page_title="Network Analytics")

st.title("🕸️ Fraud Network Analysis")
st.caption("Analisis hubungan antar Faskes, Peserta, Perusahaan, DPJP & Diagnosis")

# =============================
# Upload CSV
# =============================
# Safe CSV reader
def safe_read_csv(file):
    # Coba delimiter umum
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(file, sep=sep, engine="python", encoding="utf-8")
            if len(df.columns) > 1:
                return df
        except:
            pass

    # Coba encoding lain
    for enc in ["latin-1", "utf-16", "ISO-8859-1"]:
        try:
            df = pd.read_csv(file, sep=",", engine="python", encoding=enc)
            if len(df.columns) > 1:
                return df
        except:
            pass

    return None


uploaded = st.file_uploader("Upload file CSV klaim", type=["csv"])

if uploaded and not st.session_state.inference_done:

    st.write("File uploaded:", uploaded.name)

    df = pd.read_csv(uploaded)
    st.session_state.df_uploaded = df
    

    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

    # =============================
    # 🔥 Hanya jalankan sekali
    # =============================
    with st.spinner("Mengirim file ke backend dan memproses..."):
        files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
        
        try:
            response = requests.post(f"{API_BASE}/inference_graph", files=files)
        except Exception as e:
            st.error(f"Request error: {e}")
            st.stop()

        if response.status_code != 200:
            st.error("Server error: " + response.text)
            st.stop()

        # Simpan hasil
        result = response.json()
        predictions_url = API_BASE + "/" + result["predictions_url"]
        st.session_state.predictions_url = predictions_url
        
        df_match = pd.read_csv(predictions_url) 
        st.session_state.df_match = df_match
        
        st.session_state.inference_results = result
        st.session_state.inference_done = True

        st.success("Inference Completed! Silakan pilih faskes di bawah.")

if st.session_state.inference_done:

    df = st.session_state.df_uploaded
    predictions_url = st.session_state.predictions_url  # ✅ ambil dari session_state
    df_match =  st.session_state.get("df_match") 
    df = df.merge(
        df_match[['claim_id', 'fraud_prediction']], 
        on='claim_id', 
        how='left'
    )
    result = st.session_state.inference_results

    # Opsi untuk memilih Faskes ID
    st.subheader("🏥 Pilih Faskes untuk Analisis Risiko")

    faskes_list = df["faskes_id"].unique().tolist()
    selected_faskes = st.selectbox("Faskes ID", faskes_list)

    # Slider untuk memilih radius
    radius = st.slider("Pilih Radius untuk Subgraph", min_value=1, max_value=10, value=5, step=1)

    # Filtering lokal (tidak trigger inference lagi)
    filtered_df = df[df['faskes_id'] == selected_faskes]

    # Membangun graf klaim setelah pemilihan Faskes ID
    G = build_claim_graph(filtered_df)

    # Pilih node untuk pusat graf setelah G dibentuk
    nodes_in_graph = sorted([node for node in G.nodes])  # Mengurutkan node berdasarkan abjad
    selected_node = st.selectbox("Pilih Node untuk Pusat Graf", nodes_in_graph)

    # Render Subgraph
    if selected_faskes:
        st.subheader("🕸️ Visualisasi Graf Kolusi (Subgraph)")

        # Periksa apakah node yang dipilih ada di graf
        if selected_node in G.nodes:
            # Buat subgraf berdasarkan radius yang dipilih
            subG = nx.ego_graph(G, selected_node, radius=radius)

            # Setup Visualisasi menggunakan pyvis
            nt = Network(
                height="700px",
                width="100%",
                bgcolor="#ffffff",
                font_color="black",
                directed=False
            )
            nt.barnes_hut(gravity=-20000, central_gravity=0)
            nt.toggle_physics(False)

            # Menambahkan node ke dalam visualisasi
            for node, data in subG.nodes(data=True):
                node_type = data.get("type", "")
                fraud = data.get("fraud", 0)  # Ambil nilai fraud dari node (claim)

                # Menentukan warna dan simbol berdasarkan jenis node dan fraud
                if node_type == "faskes":
                    color, shape, size = "#ff6666", "dot", 20
                elif node_type == "participant":
                    color, shape, size = "#66b3ff", "dot", 10
                elif node_type == "dpjp":
                    color, shape, size = "#99ff99", "diamond", 14
                elif node_type == "icd":
                    color, shape, size = "#ffcc66", "triangle", 14
                elif node_type == "claim":
                    # Jika fraud_prediction == 1, set warna hitam dan bentuk simbol khusus
                    if fraud == 1:
                        color, shape, size = "black", "star", 18
                    else:
                        color, shape, size = "#ff99ff", "star", 18
                else:
                    color, shape, size = "#cccccc", "dot", 8

                nt.add_node(node, label=node, color=color, shape=shape, size=size)

            # Menambahkan edge ke dalam visualisasi
            for src, dst, data in subG.edges(data=True):
                nt.add_edge(src, dst)

            # Simpan graf dan tampilkan
            nt.save_graph("graph.html")
            st.components.v1.html(open("graph.html").read(), height=700)

    # =============================
    # LOAD RESULTS (Tidak POST ulang)
    # =============================
    predictions_url = API_BASE + "/" + result["predictions_url"]
    explanations_url = API_BASE +"/" + result["explanations_url"]
    # report_url       = API_BASE + result["report_url"]

    st.subheader("📌 Explanation Results")
    explanations = requests.get(explanations_url).json()
    # Iterasi melalui setiap item penjelasan
    for item in explanations:
        with st.expander(f"Claim ID: `{item['claim_id']}` — Prediction: **{item['prediction']}** ({item['confidence']*100:.1f}%)"):
            # Menampilkan penjelasan naratif
            st.markdown("#### 📖 Narrative Explanation")
            st.markdown(item["narrative"])
            st.markdown("---")

    # =============================
    # 3. LOAD & DISPLAY PREDICTIONS CSV
    # =============================
    st.subheader("📊 Predictions Table (CSV)")

    df = pd.read_csv(predictions_url)   # CSV bisa langsung load dari URL
    st.dataframe(df)

    st.success("Inference Completed!")

