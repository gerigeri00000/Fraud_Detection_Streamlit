# frontend/app.py
import streamlit as st
import pandas as pd
import requests
import io
from datetime import date
import base64
import altair as alt
import networkx as nx
from network_analysis import build_claim_graph, calculate_graph_risk


def load_provinces():
    url = "https://wilayah.id/api/provinces.json"
    res = requests.get(url).json()
    return res["data"]

@st.cache_data
def load_regencies(province_code):
    if not province_code:
        return []
    url = f"https://wilayah.id/api/regencies/{province_code}.json"
    res = requests.get(url).json()
    return res["data"]
 # in docker compose, backend service name
API_BASE = st.secrets["API_BASE"]

st.set_page_config(layout="wide", page_title="Fraud Triage Demo")

st.title("🔍 Fraud Detection — Single Claim Checker")

st.subheader("👥 Demographic & Geographic")

# --- PROVINSI DAN KABUPATEN DI LUAR FORM ---
provinces = load_provinces()
province_options = [""] + [prov["name"] for prov in provinces]
province_selected = st.selectbox("Provinsi", province_options)

province_code = next(
    (p["code"] for p in provinces if p["name"] == province_selected),
    None
)

regencies = load_regencies(province_code)
regency_options = [""] + [reg["name"] for reg in regencies]
regency_selected = st.selectbox("Kabupaten", regency_options)

with st.form("single_claim_form"):
    # =====================
    # 🧾 Core Identifiers
    # =====================
    st.subheader("🧾 Core Identifiers")
    col1, col2, col3 = st.columns(3)
    with col1:
        claim_id = st.text_input("Claim ID")
        episode_id = st.text_input("Episode ID")
    with col2:
        participant_id = st.text_input("Participant ID")
        nik_hash = st.text_input("NIK Hash")
        nik_hash_reuse_count = st.number_input("Berapa kali hash NIK ini digunakan dalam klaim", min_value=0, step=1)
    with col3:
        faskes_id = st.text_input("Faskes ID")
        dpjp_id = st.text_input("DPJP ID")

    # =====================
    # 👥 Demographic & Geographic
    # =====================
    st.subheader("👥 Demographic & Geographic")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0, step=1)
    with col2:
        sex = st.selectbox("Sex", ["", "M", "F"])
        provinsi = province_selected
        kabupaten = regency_selected
    with col3:
        faskes_level = st.selectbox("Faskes Level", ["", "FKTP", "FKRTL"])

    # =====================
    # ⚕️ Clinical Information
    # =====================
    st.subheader("⚕️ Clinical Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        tgl_pelayanan = st.date_input("Tanggal Pelayanan", value=None)
        kode_icd10 = st.text_input("Kode ICD-10")
        time_diff_prev_claim = st.number_input("Berapa Hari Sejak Klaim Sebelumnya?", min_value=0, step=1)
    with col2:
        kode_prosedur = st.text_input("Kode Prosedur")
        jenis_pelayanan = st.selectbox("Jenis Pelayanan", ["", "Rawat Jalan", "Rawat Inap"])
        claim_month = st.number_input("Bulan Klaim (1-12)", min_value=1, max_value=12, step=1)
        claim_quarter = (claim_month - 1) // 3 + 1
    with col3:
        room_class = st.selectbox("Kelas Rawat", ["", "Kelas 1", "Kelas 2", "Kelas 3", "VIP"])
        lama_dirawat = st.number_input("Lama Dirawat (hari)", min_value=0, step=1)

    # =====================
    # 💰 Financial Data
    # =====================
    st.subheader("💰 Financial Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        billed_amount = st.number_input("Billed Amount (Rp)", min_value=0)
        paid_amount = st.number_input("Paid Amount (Rp)", min_value=0)
        tarif_inacbg = st.number_input("Tarif INACBG (Rp)", min_value=0)
    with col2:
        drug_cost = st.number_input("Drug Cost (Rp)", min_value=0)
        procedure_cost = st.number_input("Procedure Cost (Rp)", min_value=0)
        rolling_avg_cost_30d: int = st.number_input("Rata-rata biaya klaim dalam 30 hari terakhir (Rp)", min_value=0)
    with col3:
        selisih_klaim = billed_amount - paid_amount
        provider_monthly_claims = st.number_input("Total Klaim Provider dalam Sebulan (Rp)", min_value=0)
        st.text_input("Selisih Klaim (auto)", value=str(selisih_klaim), disabled=True)

    # =====================
    # 📊 Behavioral Metrics
    # =====================
    st.subheader("📊 Behavioral Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        visit_count_30d = st.number_input("Visit Count (Last 30 days)", min_value=0)
        clinical_pathway_deviation_score = st.number_input("Skor Deviasi Jalur Klinis", min_value=0.0, max_value=1.0, step=0.01)
    with col2:
        kapitasi_flag = st.checkbox("Kapitasi Flag (Capitation Payment)?")
        referral_flag = st.checkbox("Referral Flag (Has Referral)?")
    with col3:
        referral_to_same_facility = st.checkbox("Referral to Same Facility?")

    # =====================
    # 🧮 Computed Features
    # =====================
    claim_ratio = billed_amount / paid_amount if paid_amount else 0
    drug_ratio = drug_cost / billed_amount if billed_amount else 0
    procedure_ratio = procedure_cost / billed_amount if billed_amount else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("Claim Ratio (auto)", value=f"{claim_ratio:.2f}", disabled=True)
    with col2:
        st.text_input("Drug Ratio (auto)", value=f"{drug_ratio:.2f}", disabled=True)
    with col3:
        st.text_input("Procedure Ratio (auto)", value=f"{procedure_ratio:.2f}", disabled=True)

    provider_claim_share = st.number_input("Provider Claim Share", min_value=0.0, max_value=1.0, step=0.01)

    submitted = st.form_submit_button("🔎 Check Risk")

if submitted:
    payload = {
        # Core Identifiers
        "claim_id": claim_id,
        "episode_id": episode_id,
        "participant_id": participant_id,
        "nik_hash": nik_hash,
        "faskes_id": faskes_id,
        "dpjp_id": dpjp_id,
        "nik_hash_reuse_count": 1,

        # Demographic & Geographic
        "age": age,
        "sex": sex,
        "provinsi": provinsi,
        "kabupaten": kabupaten,
        "faskes_level": faskes_level,

        # Clinical Information
        "tgl_pelayanan": tgl_pelayanan.isoformat(),
        "kode_icd10": kode_icd10,
        "kode_prosedur": kode_prosedur,
        "jenis_pelayanan": jenis_pelayanan,
        "room_class": room_class,
        "lama_dirawat": lama_dirawat,
        "time_diff_prev_claim": time_diff_prev_claim,
        

        # Financial Data
        "billed_amount": billed_amount,
        "paid_amount": paid_amount,
        "selisih_klaim": selisih_klaim,
        "drug_cost": drug_cost,
        "procedure_cost": procedure_cost,
        "tarif_inacbg": tarif_inacbg,
        "rolling_avg_cost_30d": rolling_avg_cost_30d,
        "provider_monthly_claims": provider_monthly_claims,
        "claim_month" : claim_month,
        'claim_quarter': claim_quarter,

        # Behavioral Metrics
        "clinical_pathway_deviation_score": clinical_pathway_deviation_score,
        "visit_count_30d": visit_count_30d,
        "kapitasi_flag": kapitasi_flag,
        "referral_flag": referral_flag,
        "referral_to_same_facility": referral_to_same_facility,
        # "fraud_flag": 0,
        # "fraud_type": "upcoding_diagnosis",

        # Computed Features
        "claim_ratio": claim_ratio,
        "drug_ratio": drug_ratio,
        "procedure_ratio": procedure_ratio,
        "provider_claim_share": provider_claim_share,
    }
    
    with st.spinner("Evaluating risk..."):
        response = requests.post(f"{API_BASE}/score_single", json=payload)

    if response.status_code == 200:
        predictions = response.json()["predictions"]
        evaluation = response.json()["evaluation"]
        st.json(predictions)
        st.json(evaluation)
        # print("PREDICTIONS:", predictions)
        # print("EVALUATION:", evaluation)
        # result = response.json()["result"]
        

        # # HEADER
        # st.markdown(f"## 🧾 Hasil Prediksi Klaim — **{result['claim_id']}**")

        # # MAIN CARD
        # risk_color = {
        #     "GREEN": "🟩 Green",
        #     "AMBER": "🟨 Amber",
        #     "RED":   "🟥 Red"
        # }.get(result["label"], result["label"])

        # st.markdown(
        #     f"""
        #     <div style="padding:20px; border-radius:12px; border:1px solid #DDD;">
        #         <h3>📊 Risk Assessment</h3>
        #         <p><b>Risk Score:</b> {result['risk_score']:.3f}</p>
        #         <p><b>Label:</b> {risk_color}</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )

        # st.markdown("---")

        # # REASONS LIST
        # st.markdown("### 🔍 Alasan Terdeteksi")
        
        # st.markdown(
        #             f"""
        #             <div style="
        #                 padding:20px;
        #                 border-radius:12px;
        #                 border:1px solid #DDD;
        #                 margin-bottom:16px;
        #             ">
        #                 <h3 style="margin-top:0;">🧠 Generative AI</h3>
        #                 <p><b>Ini Tempat Nanti Genarative AI Menjelaskan... (fitur dalam pengembangan)</b></p>
        #             </div>
        #             """,
        #             unsafe_allow_html=True
        #         )

        # if result["reasons"]:
        #     for r in result["reasons"]:
        #         st.markdown(
        #             f"""
        #             <div style="
        #                 padding:20px;
        #                 border-radius:12px;
        #                 border:1px solid #DDD;
        #                 margin-bottom:16px;
        #             ">
        #                 <h3 style="margin-top:0;">🧠 Rule Triggered</h3>
        #                 <p><b>Rule Name:</b> {r['rule']}</p>
        #                 <p><b>Explanation:</b> {r['explanation']}</p>
        #             </div>
        #             """,
        #             unsafe_allow_html=True
        #         )
        #         # Bagian generative AI tambahan
        #     with st.expander("💡 Penjelasan AI (Generative Explanation)"):
        #         user_prompt = f"Jelaskan dengan bahasa sederhana mengapa rule berikut terpicu: {r['explanation']}"
        #         #ai_response = generate_ai_explanation(user_prompt)   # fungsi API buatan kamu

        #         st.write("Ini Tempat Nanti Genarative AI Menjelaskan... (fitur dalam pengembangan)")
        # else:
        #     st.success("Tidak ada rules yang terpicu — klaim tampak normal.")


        # # RAW JSON (hide inside expander)
        # with st.expander("📦 Raw JSON Result"):
        #     st.json(result)

    else:
        st.error("❌ Failed to get prediction.")

st.title("📊 Batch Scoring Dashboard")

uploaded = st.file_uploader("Upload scored parquet/csv (or use demo)", type=["parquet","csv"])

if uploaded:
    with st.spinner("⏳ Mengirim ke backend untuk scoring..."):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        response = requests.post(f"{API_BASE}/batch_score", files=files)

    if response.status_code != 200:
        st.error("Gagal memproses batch.")
        st.stop()
    
    else:
        data = response.json()
        csv_b64 = data["predictions_csv_b64"]
        df = pd.read_csv(io.BytesIO(base64.b64decode(csv_b64)))
        st.subheader("📋 Prediction Summary")

        # 1️⃣ Hitung total baris
        total_rows = len(df)

        # 2️⃣ Hitung total kolom predicted_fraud == 1
        total_fraud_predicted = df["predicted_fraud"].sum()

        # 3️⃣ Hitung jumlah per kategori predicted_fraud_type
        fraud_type_counts = df["predicted_fraud_type"].value_counts()
            
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total Rows", f"{total_rows:,}")
        col2.metric("Predicted Fraud (1)", f"{total_fraud_predicted:,}")
        col3.metric("Predicted Not Fraud (0)", f"{(total_rows - total_fraud_predicted):,}")

        # Hapus "benign"
        fraud_only = fraud_type_counts.drop("benign", errors="ignore")
        df_chart = fraud_only.reset_index()
        df_chart.columns = ["fraud_type", "count"]

        chart = (
            alt.Chart(df_chart)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Jumlah"),
                y=alt.Y("fraud_type:N", sort="-x", title="Fraud Type"),
                tooltip=["fraud_type", "count"]
            )
            .properties(width=600, height=450)
        )

        st.subheader("📊 Fraud Type Distribution")
        st.altair_chart(chart, use_container_width=True)
        
        
        st.markdown("---")
        

        st.subheader("🔍 Predicted Fraud Type Breakdown")
        st.write(fraud_type_counts)
        
        # === DOWNLOAD BUTTON ===
        st.subheader("📥 Download Predictions")
        csv_bytes = base64.b64decode(csv_b64)

        st.download_button(
            "Download Predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
        

else:
    # try ask backend for sample via / (or show message)
    st.info("Upload scored file exported from backend, or run demo seeds.")
    df = pd.DataFrame()
