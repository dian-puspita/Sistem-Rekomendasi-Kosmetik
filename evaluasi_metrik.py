import numpy as np
import pandas as pd

def precision_at_k(recommended_ids, relevant_ids, k):
    """
    Precision@k yang disesuaikan
    Jika ground truth lebih kecil dari k, tetap pakai k
    """
    if not relevant_ids:
        return 0.0
    recommended_k = recommended_ids[:k]
    return len(set(recommended_k) & set(relevant_ids)) / min(k, len(recommended_ids))

def recall_at_k(recommended_ids, relevant_ids, k):
    """
    Recall@k normal, tetap membandingkan dengan jumlah relevan sebenarnya
    """
    if not relevant_ids:
        return 0.0
    recommended_k = recommended_ids[:k]
    return len(set(recommended_k) & set(relevant_ids)) / len(relevant_ids)

def ndcg_at_k(recommended_ids, relevant_ids, k):
    """
    NDCG@k yang disesuaikan
    Menghitung DCG dan ideal DCG hanya sampai min(k, len(recommended_ids))
    """
    if not relevant_ids:
        return 0.0

    recommended_k = recommended_ids[:k]
    dcg = 0.0
    for i, pid in enumerate(recommended_k):
        if pid in relevant_ids:
            dcg += 1 / np.log2(i + 2)  # posisi dimulai dari 0

    ideal_k = min(len(relevant_ids), k)
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(ideal_k))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
