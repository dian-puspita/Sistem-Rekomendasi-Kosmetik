import streamlit as st
import pandas as pd
from modules.user_filter import filter_user_preferences
from modules.image import generate_dummy_image
from modules.recommendation import hybrid_topk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from PIL import Image

st.set_page_config(page_title="Skincare Recommendation", layout="wide")

# ====== Load dataset ======
CSV_PATH = "data/skincare_products_clean.csv"
df = pd.read_csv(CSV_PATH)

# --- Normalisasi teks ---
for col in ['Category', 'Skin_Type', 'Gender', 'Usage_Frequency', 'Product_Name', 'Brand', 'Ingredients']:
    if col in df.columns:
        df[col] = df[col].fillna('').astype(str).str.strip().str.lower()
    else:
        df[col] = ''

# Price numeric safety
if 'Price_IDR' in df.columns:
    df['Price_IDR'] = pd.to_numeric(df['Price_IDR'], errors='coerce').fillna(0)
else:
    df['Price_IDR'] = 0

# ===== Precompute TF-IDF sekali =====
st.session_state.setdefault("tfidf_matrix", None)
st.session_state.setdefault("tfidf_vectorizer", None)

if st.session_state.tfidf_matrix is None:
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = csr_matrix(tfidf_vectorizer.fit_transform(df['combined_features']))
    st.session_state.tfidf_matrix = tfidf_matrix
    st.session_state.tfidf_vectorizer = tfidf_vectorizer
else:
    tfidf_matrix = st.session_state.tfidf_matrix

# ----- Halaman & Sidebar Styling -----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
    font-size: 15px;
    color: #0b3d91;
}

/* Title & Headers */
h1 {
    font-size: 34px !important;
    font-weight: 600;
}
h2 {
    font-size: 26px !important;
}
h3 {
    font-size: 20px !important;
}
h4 {
    font-size: 18px !important;
}
p, span, div, label {
    font-size: 15px;
}

/* Background halaman utama */
.reportview-container, .main, .block-container {
    background: linear-gradient(135deg, #dbe6f1, #a8c0ff);
    color: #0b3d91;
}

/* Sidebar */
.sidebar .sidebar-content {
    background-color: #647d9c;
    padding: 20px;
    border-radius: 12px;
    color: #ffffff;
}
.sidebar .sidebar-content label,
.sidebar .sidebar-content div {
    color: #ffffff;
    font-size: 15px;
}
.sidebar .sidebar-content h2 {
    font-size: 18px !important;
    color: #ffffff;
}

/* Card styling */
.card {
    background: linear-gradient(135deg, #f1f5f9, #c0d4e3);
    border-radius: 12px;
    padding: 15px;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.25);
    transition: transform 0.2s;
    margin-bottom: 15px;
    font-size: 15px;
}
.card:hover {
    transform: scale(1.03);
}

/* Product title in card */
.card h3 {
    font-size: 20px !important;
    font-weight: 600;
    margin: 0;
}

/* Tombol */
.stButton>button {
    background-color: #0b3d91;
    color: #ffffff;
    border-radius: 8px;
    font-size: 15px;
}
.stButton>button:hover {
    background-color: #0f4ba4;
}
</style>
""", unsafe_allow_html=True)

st.title("🏆Cosmetics Recommendation System")
st.sidebar.header("Product Filters")

# ----- Ambil unique value -----
categories = ["All"] + sorted(df['Category'].dropna().unique().tolist())
skin_types = ["All"] + sorted(df['Skin_Type'].dropna().unique().tolist())
genders = ["All"] + sorted(df['Gender'].dropna().unique().tolist())
usage_freqs = ["All"] + sorted(df['Usage_Frequency'].dropna().unique().tolist())

# ----- Filter Sidebar -----
category_select = st.sidebar.selectbox("Product Categories", categories)
skin_type_select = st.sidebar.selectbox("Skin Type", skin_types)
gender_select = st.sidebar.selectbox("Gender", genders)
usage_select = st.sidebar.selectbox("Usage Frequency", usage_freqs)

# nilai MIN dan MAX dari kolom Price_IDR
price_min = float(df['Price_IDR'].min()) if len(df) > 0 else 0.0
price_max = float(df['Price_IDR'].max()) if len(df) > 0 else 0.0

st.sidebar.markdown("Product Price Range (Rp)")
price_range = st.sidebar.slider(
    "Select price range:",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
    step=5000.0,
    format="%.0f"
)
st.sidebar.caption(f"Displaying products with prices between **Rp {price_range[0]:,.0f}** and **Rp {price_range[1]:,.0f}**")

# ----- Convert "All" => None -----
category = None if category_select == "All" else category_select
skin_type = None if skin_type_select == "All" else skin_type_select
gender = None if gender_select == "All" else gender_select
usage_frequency = None if usage_select == "All" else usage_select

# ----- Apply filter -----
filtered_df = filter_user_preferences(df, category, skin_type, gender, usage_frequency, price_range)

# ----- Tentukan jumlah top recommendation -----
top_k = 20 if all(x is None for x in [category, skin_type, gender, usage_frequency]) else 10

# ----- Hitung rekomendasi -----
top_recommendations = hybrid_topk(
    filtered_df,
    alpha=0.4,
    beta=0.2,
    gamma=0.4,
    k=top_k,
    tfidf_matrix=tfidf_matrix
)

if filtered_df.empty:
    st.warning("Sorry, there are no products matching your filter.")
else:
    st.write(f"Top-{top_k} Recommended Cosmetic Products")
    for _, row in top_recommendations.iterrows():
        with st.container():
            cols = st.columns([1, 2])
            with cols[0]:
                product_name = str(row.get("Product_Name", "Unknown Product"))
                # Pakai dummy image otomatis
                img_path = generate_dummy_image(product_name)
                img = Image.open(img_path)
                st.image(img, width=180, caption=f"{row['Brand'].title()}")

            with cols[1]:
                st.markdown(f"""
                    <div class='card'>
                        <h3 style='margin:0; color:#0b3d91;'>{row['Product_Name'].title()}</h3>
                        <p style='margin:0.3em 0; font-weight:bold; color:#1a1a1a;'>Brand: {row['Brand'].title()}</p>
                        <p style='margin:0.3em 0;'><span style="background-color:#6c91c2; color:#fff; padding:3px 6px; border-radius:5px;">{row['Category'].title()}</span></p>
                        <p style='margin:0.3em 0;'><b>Rating:</b> {'⭐'*int(row['Rating'])} ({row['Rating']})</p>
                        <p style='margin:0.3em 0;'><b>Reviews:</b> {int(row['Number_of_Reviews'])}</p>
                        <p style='margin:0.3em 0; color:#0b3d91; font-size:18px;'><b>Price:</b> Rp {row['Price_IDR']:,.0f}</p>
                        <p style='margin:0.3em 0;'><b>Ingredients:</b> {row['Ingredients']}</p>
                        <p style='margin:0.3em 0;'><b>Origin:</b> {row['Country_of_Origin']}</p>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
