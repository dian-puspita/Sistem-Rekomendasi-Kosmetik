import pandas as pd
import numpy as np
from modules.recommendation import hybrid_topk
from evaluasi_metrik import precision_at_k, recall_at_k, ndcg_at_k
from modules.user_filter import filter_user_preferences
from sklearn.feature_extraction.text import TfidfVectorizer

def evaluate_hybrid_topk(df, tfidf_matrix, alpha=0.4, beta=0.2, gamma=0.4, k=10, user_index=None):
    """
    Evaluasi internal sistem rekomendasi hybrid_topk.
    
    Ground truth dibuat dari weighted_score subset (rating+popularity),
    tanpa similarity, sehingga subset filtered tidak otomatis 100% cocok.
    """
    if df.empty or len(df) < 1:
        return 0.0, 0.0, 0.0, [], []

    data = df.copy()

    # Normalisasi rating dan popularity
    data['Rating_norm'] = data['Rating'] / 5
    max_reviews = data['Number_of_Reviews'].max()
    data['Popularity_norm'] = data['Number_of_Reviews'] / max_reviews if max_reviews > 0 else 0

    # Ground truth internal (tanpa similarity)
    data['weighted_score'] = alpha*data['Rating_norm'] + beta*data['Popularity_norm'] + 0*gamma
    topk_ground_truth = data.sort_values('weighted_score', ascending=False).head(k)['Product_Name'].tolist()

    # Top-k rekomendasi hybrid
    topk_recom_df = hybrid_topk(df, tfidf_matrix, alpha=alpha, beta=beta, gamma=gamma, k=k, user_index=user_index)
    topk_recom = topk_recom_df['Product_Name'].tolist()

    # Precision, Recall, NDCG
    precision = precision_at_k(topk_recom, topk_ground_truth, k)
    recall    = recall_at_k(topk_recom, topk_ground_truth, k)
    ndcg      = ndcg_at_k(topk_recom, topk_ground_truth, k)

    return precision, recall, ndcg, topk_ground_truth, topk_recom

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/skincare_products_clean.csv")

    # Normalisasi teks
    for col in ['Category','Skin_Type','Gender','Usage_Frequency','Product_Name','Brand','Ingredients']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.lower().str.strip()
        else:
            df[col] = ''

    # Price numeric
    if 'Price_IDR' in df.columns:
        df['Price_IDR'] = pd.to_numeric(df['Price_IDR'], errors='coerce').fillna(0)
    else:
        df['Price_IDR'] = 0

    # TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    alpha, beta, gamma = 0.4, 0.2, 0.4

    # ===== Evaluasi Global Top-20 =====
    top_k_global = 20
    precision_g, recall_g, ndcg_g, top_sys_g, top_recom_g = evaluate_hybrid_topk(
        df, tfidf_matrix, alpha, beta, gamma, k=top_k_global
    )
    print("\n=== Evaluasi Global Top-20 ===")
    print(f"Precision@{top_k_global}: {precision_g:.4f}")
    print(f"Recall@{top_k_global}: {recall_g:.4f}")
    print(f"NDCG@{top_k_global}: {ndcg_g:.4f}")
    print(f"Top-20 Ground Truth (rating+popularity): {top_sys_g}")
    print(f"Top-20 Rekomendasi Hybrid: {top_recom_g}")

    # ===== Evaluasi Filtered Top-10 =====
    # Tentukan kombinasi filter yang realistis
    filter_options = [
        {"category":"blush","skin_type":"oily","gender":"female","usage_frequency":"weekly"},
        {"category":"skincare","skin_type":"dry","gender":"female","usage_frequency":"daily"},
        {"category":"makeup","skin_type":None,"gender":None,"usage_frequency":None}, # filter sebagian kosong
        {"category":None,"skin_type":"combination","gender":"male","usage_frequency":"monthly"},
        {"category":None,"skin_type":None,"gender":None,"usage_frequency":None}, # filter semua kosong
    ]

    all_precisions = []
    all_recalls = []
    all_ndcgs = []

    for f in filter_options:
        filtered_df = filter_user_preferences(
            df,
            category=f.get("category"),
            skin_type=f.get("skin_type"),
            gender=f.get("gender"),
            usage_frequency=f.get("usage_frequency"),
            price_range=(0, df['Price_IDR'].max())
        )

        if len(filtered_df) < 1:
            continue

        # batasi top_k jika produk kurang dari k
        k_actual = min(10, len(filtered_df))
        precision, recall, ndcg, _, _ = evaluate_hybrid_topk(filtered_df, tfidf_matrix,
                                                            alpha, beta, gamma, k=k_actual)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_ndcgs.append(ndcg)

    # Rata-rata metrik untuk semua kombinasi filter
    avg_precision = sum(all_precisions)/len(all_precisions)
    avg_recall    = sum(all_recalls)/len(all_recalls)
    avg_ndcg      = sum(all_ndcgs)/len(all_ndcgs)

    print("\n=== Average Metrics Across Multiple Filters ===")
    print(f"Precision@10: {avg_precision:.4f}")
    print(f"Recall@10: {avg_recall:.4f}")
    print(f"NDCG@10: {avg_ndcg:.4f}")
