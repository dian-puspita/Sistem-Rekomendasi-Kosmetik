from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

"""
    Sistem rekomendasi hybrid top-k berbasis rating, popularitas, dan similarity TF-IDF.

    Parameters:
        df           : DataFrame subset/full dataset
        tfidf_matrix : TF-IDF matrix seluruh dataset
        alpha, beta, gamma : bobot untuk rating, popularitas, similarity
        k            : jumlah produk teratas yang ingin direkomendasikan
        user_index   : index global produk referensi (opsional), untuk preferensi user

    Returns:
        DataFrame top-k produk dengan kolom weighted_score
"""

def hybrid_topk(df, tfidf_matrix, alpha=0.4, beta=0.2, gamma=0.4, k=10, user_index=None):

    if df.empty:
        return pd.DataFrame()
    
    data = df.copy()
    
    # --- Normalisasi rating dan popularity ---
    # Rating dibagi 5 agar dalam range [0,1]
    data['Rating_norm'] = data['Rating'] / 5

    # Popularitas berdasarkan jumlah review, dinormalisasi dengan max review
    max_reviews = data['Number_of_Reviews'].max()
    data['Popularity_norm'] = data['Number_of_Reviews'] / max_reviews if max_reviews > 0 else 0

    # --- Similarity ---
    if user_index is not None:
        # mapping user_index global ke index lokal filtered df
        if user_index in df.index:
            local_idx = df.index.get_loc(user_index)
            tfidf_subset = tfidf_matrix[df.index, :]
            sim_scores = cosine_similarity(tfidf_subset[local_idx], tfidf_subset).flatten()
        else:
            # jika user_index tidak ada di subset, pakai mean similarity
            tfidf_subset = tfidf_matrix[df.index, :]
            sim_scores = cosine_similarity(tfidf_subset, tfidf_subset).mean(axis=1)
    else:
        tfidf_subset = tfidf_matrix[df.index, :]
        sim_scores = cosine_similarity(tfidf_subset, tfidf_subset).mean(axis=1)

    # Normalisasi similarity
    data['Similarity_norm'] = sim_scores

    # --- Weighted score ---
    # Menggabungkan rating, popularitas, dan similarity sesuai bobot
    data['weighted_score'] = (
        alpha * data['Rating_norm'] +
        beta  * data['Popularity_norm'] +
        gamma * data['Similarity_norm']
    )

    # --- Ambil top-k produk berdasarkan weighted_score ---
    return data.sort_values('weighted_score', ascending=False).head(k)
