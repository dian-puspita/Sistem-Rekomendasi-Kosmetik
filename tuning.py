import pandas as pd
import numpy as np
from modules.recommendation import hybrid_topk
from modules.precompute_similarity import build_tfidf_matrix
from modules.user_filter import filter_user_preferences
from evaluasi_metrik import precision_at_k, recall_at_k, ndcg_at_k

# ===== Load dataset =====
df = pd.read_csv("data/skincare_products_clean.csv")

# ===== Build TF-IDF matrix =====
tfidf_matrix = build_tfidf_matrix(df)

# ===== Parameter grid untuk tuning weighted_score =====
ALPHA_LIST = [0.1, 0.2, 0.3, 0.4]   # rating
BETA_LIST  = [0.1, 0.2, 0.3, 0.4]   # popularity
GAMMA_LIST = [0.4, 0.5, 0.6, 0.7]   # similarity
K = 10  # top-k untuk evaluasi

results = []

print("=== START INTERNAL FAST TUNING α β γ ===\n")

for alpha in ALPHA_LIST:
    for beta in BETA_LIST:
        for gamma in GAMMA_LIST:
            # ===== Buat weighted score ground truth internal =====
            data = df.copy()
            data['Rating_norm'] = data['Rating'] / 5
            max_reviews = data['Number_of_Reviews'].max()
            data['Popularity_norm'] = data['Number_of_Reviews'] / max_reviews if max_reviews > 0 else 0
            data['weighted_score'] = alpha*data['Rating_norm'] + beta*data['Popularity_norm'] + gamma*0  # similarity=0 untuk ground truth
            topk_system = data.sort_values('weighted_score', ascending=False).head(K)['Product_Name'].tolist()

            # ===== Rekomendasi hybrid_topk =====
            topk_recom = hybrid_topk(df, tfidf_matrix, alpha=alpha, beta=beta, gamma=gamma, k=K, user_index=None)['Product_Name'].tolist()

            # ===== Hitung metrik internal =====
            precision = precision_at_k(topk_recom, topk_system, K)
            recall    = recall_at_k(topk_recom, topk_system, K)
            ndcg      = ndcg_at_k(topk_recom, topk_system, K)

            results.append({
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg
            })

            print(f"α={alpha}, β={beta}, γ={gamma} -> Precision={precision:.3f}, Recall={recall:.3f}, NDCG={ndcg:.3f}")

# ===== Simpan hasil tuning =====
df_result = pd.DataFrame(results)
df_result.to_csv("tuning_internal_results.csv", index=False)
print("\n=== FINISHED: tuning_internal_results.csv created ===")
