import streamlit as st
import pandas as pd
from pyvis.network import Network
import networkx as nx
from network_analysis import build_claim_graph, calculate_graph_risk
import requests
import json

API_BASE = st.secrets["API_BASE"]
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
if uploaded:
    st.write("File uploaded:", uploaded.name)
    df = pd.read_csv(uploaded)
    if df is not None:
        st.subheader("📄 Preview Data")
        st.dataframe(df.head())

        # Build graph

        # =============================
        # Select Faskes to Inspect
        # =============================
        st.subheader("🏥 Pilih Faskes untuk Analisis Risiko")
        faskes_list = df["faskes_id"].unique().tolist()
        selected_faskes = st.selectbox("Faskes ID", faskes_list)
        filtered_df = df[df['faskes_id'] == selected_faskes]
        G = build_claim_graph(filtered_df)
        if selected_faskes:
            st.subheader("🕸️ Visualisasi Graf Kolusi (Subgraph)")

            selected_node = f"FSK_{selected_faskes}"

            if selected_node in G.nodes:

                subG = nx.ego_graph(G, selected_node, radius=2)

                nt = Network(
                    height="700px",
                    width="100%",
                    bgcolor="#ffffff",
                    font_color="black",
                    directed=False
                )

                # Matikan physics → graph diam & stabil
                nt.barnes_hut(gravity=-20000, central_gravity=0)
                nt.toggle_physics(False)

                # Tambahkan node dengan warna sesuai tipe
                for node, data in subG.nodes(data=True):
                    node_type = data.get("type", "")

                    if node_type == "faskes":
                        color = "#ff6666"   # merah
                        shape = "dot"
                        size = 20
                    elif node_type == "participant":
                        color = "#66b3ff"   # biru muda
                        shape = "dot"
                        size = 10
                    elif node_type == "dpjp":
                        color = "#99ff99"   # hijau
                        shape = "diamond"
                        size = 14
                    elif node_type == "icd":
                        color = "#ffcc66"   # kuning
                        shape = "triangle"
                        size = 14
                    else:
                        color = "#cccccc"
                        shape = "dot"
                        size = 8

                    nt.add_node(node, label=node, color=color, shape=shape, size=size)

                # Tambahkan edge
                for src, dst, data in subG.edges(data=True):
                    nt.add_edge(src, dst)

                # Render
                nt.save_graph("graph.html")
                st.components.v1.html(open("graph.html", "r").read(), height=700)

            else:
                st.warning("Faskes tidak ditemukan dalam graf.")
        # selected_node = f"FSK_{selected_faskes}"
        # subG = nx.ego_graph(G, selected_node, radius=2)
        
        # # =============================
        # # Graph Visualization
        # # =============================
        # st.subheader("🕸️ Visualisasi Graf Kolusi")
        # nt = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        # nt.from_nx(subG)
        # nt.save_graph("graph.html")
        # st.components.v1.html(open("graph.html", "r").read(), height=600)
    else:
        st.error("❌ CSV tidak bisa dibaca. Pastikan file CSV valid dan menggunakan delimiter koma / titik koma.")
        st.stop()

    if st.button("Proses Inference"):
        with st.spinner("Mengirim file ke backend dan memproses..."):
            # kirim file dalam multipart/form-data
            files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
            try:
                response = requests.post(f"{API_BASE}/inference_graph", files=files)
            except Exception as e:
                st.error(f"Request error: {e}")
                st.stop()
                
            if response.status_code != 200:
                st.error("Server error: " + response.text)
                st.stop()
                
            result = response.json()
            print(result)

            predictions_url = API_BASE + result["predictions_url"]
            explanations_url = API_BASE + result["explanations_url"]
            report_url = API_BASE + result["report_url"]

            # =============================
            # 1. LOAD & DISPLAY EXPLANATIONS (JSON)
            # =============================
            st.subheader("📌 Explanation Results")
            exp_response = requests.get(explanations_url)
            explanations = exp_response.json()

            for item in explanations:
                st.markdown(f"### Claim ID: `{item['claim_id']}` — Prediction: **{item['prediction']}** ({item['confidence']*100:.1f}%)")
                
                st.markdown("#### 🧠 Reasoning Trace")
                for step in item["schema"]["reasoning_trace"]:
                    st.markdown(f"- {step}")

                st.markdown("#### 🔗 Important Edges")
                st.json(item["schema"]["explainer_results"]["important_edges"])

                st.markdown("#### 🧩 Important Features")
                st.json(item["schema"]["explainer_results"]["important_features"])

                st.markdown("#### 📖 Narrative Explanation")
                st.markdown(item["narrative"])

                st.markdown("---")


            # =============================
            # 2. LOAD & DISPLAY REPORT (TXT)
            # =============================
            report_response = requests.get(report_url)
            report_text = report_response.text

            st.text_area("Report", report_text, height=400)

            # =============================
            # 3. LOAD & DISPLAY PREDICTIONS CSV
            # =============================
            st.subheader("📊 Predictions Table (CSV)")

            df = pd.read_csv(predictions_url)   # CSV bisa langsung load dari URL
            st.dataframe(df)

            st.success("Inference Completed!")
    
    
    
