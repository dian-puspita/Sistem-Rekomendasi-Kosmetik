from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# Membangun TF-IDF matrix dari kolom 'combined_features' dataframe.
def build_tfidf_matrix(df):

    # Inisialisasi TF-IDF vectorizer, stop_words='english' agar kata umum seperti 'and', 'the' diabaikan
    tfidf = TfidfVectorizer(stop_words='english')

    # Fit dan transform teks menjadi TF-IDF matrix
    matrix = tfidf.fit_transform(df['combined_features'])
    
    # Konversi ke format sparse agar hemat memori dan cepat untuk similarity
    return csr_matrix(matrix)
