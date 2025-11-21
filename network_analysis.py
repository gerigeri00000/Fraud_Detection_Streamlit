import pandas as pd
import networkx as nx

# ============================
# BUILD GRAPH
# ============================
def build_claim_graph(df: pd.DataFrame):
    G = nx.Graph()

    # Pastikan tidak ada nilai kosong di kolom yang digunakan
    required_columns = ['claim_id', 'participant_id', 'faskes_id', 'dpjp_id', 'kode_icd10']
    if any(df[col].isnull().any() for col in required_columns):
        raise ValueError("Dataframe memiliki nilai kosong di salah satu kolom yang dibutuhkan")

    # Iterasi untuk menambahkan node dan edge ke graf
    for _, row in df.iterrows():
        claim_id = row['claim_id']
        participant = f"PTC_{row['participant_id']}"
        faskes = f"FSK_{row['faskes_id']}"
        dpjp = f"DR_{row['dpjp_id']}"
        icd = f"ICD_{row['kode_icd10']}"
        fraud_prediction = row['fraud_prediction']  # Ambil nilai fraud_prediction

        # Tambahkan node ke graf
        G.add_node(claim_id, type="claim", fraud=fraud_prediction)
        G.add_node(participant, type="participant")
        G.add_node(faskes, type="faskes")
        G.add_node(dpjp, type="dpjp")
        G.add_node(icd, type="icd")

        # Tambahkan edge ke graf
        G.add_edge(faskes, dpjp, relation="doctor_in_charge")
        G.add_edge(claim_id, participant, relation="filed_by")
        G.add_edge(claim_id, dpjp, relation="attended_by")
        G.add_edge(claim_id, icd, relation="diagnosis")

    # Cek jika graf terbentuk dengan benar
    if G.number_of_nodes() == 0:
        raise ValueError("Graf tidak memiliki node setelah proses pembangunan.")

    return G


# ============================
# CALCULATE RISK SCORE
# ============================
def calculate_graph_risk(G: nx.Graph, faskes_id: str):

    faskes_node = f"FSK_{faskes_id}"

    if faskes_node not in G.nodes:
        return None

    # Centrality
    bet = nx.betweenness_centrality(G).get(faskes_node, 0)
    deg = nx.degree_centrality(G).get(faskes_node, 0)

    # Community detection (Louvain alternative)
    communities = nx.algorithms.community.greedy_modularity_communities(G)

    community_score = 0
    for c in communities:
        if faskes_node in c:
            size = len(c)
            if size > 7:  # cluster besar mencurigakan
                community_score = 1
            elif size > 4:
                community_score = 0.7
            else:
                community_score = 0.3

    # Final score (0–100)
    final_score = (
        0.4 * bet +
        0.3 * deg +
        0.3 * community_score
    ) * 100

    return {
        "betweenness": bet,
        "degree": deg,
        "community_score": community_score,
        "final_risk": round(final_score, 2)
    }
