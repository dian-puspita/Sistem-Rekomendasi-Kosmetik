# 💄 Skincare Recommendation System

*Project Tugas Besar – Kecerdasan Buatan*

# 📌 Deskripsi Proyek

Sistem ini adalah **Aplikasi Rekomendasi Produk Skincare** berbasis **Hybrid Recommendation System**, yang menggabungkan:

* **Content-Based Filtering** (TF-IDF + Cosine Similarity)
* **Rating**
* **Popularitas (jumlah review)**

Pengguna dapat memilih preferensi seperti:

* Jenis produk
* Tipe kulit
* Gender
* Frekuensi penggunaan
* Rentang harga

Kemudian sistem akan memberikan **daftar rekomendasi terbaik** sesuai kebutuhan pengguna.

Aplikasi dibangun menggunakan:

* Python
* Streamlit
* Scikit-learn
* Pandas
* Pillow (PIL)

---

# 🎯 Tujuan Proyek

1. Membangun sistem rekomendasi kosmetik berbasis data.
2. Mengimplementasikan pendekatan Content-Based Filtering & Hybrid.
3. Menyediakan aplikasi rekomendasi interaktif dengan Streamlit.
4. Menyediakan metrik evaluasi rekomendasi secara kuantitatif.

---

# 🧠 Pendekatan AI yang Digunakan

## 1️⃣ Content-Based Filtering

Fitur teks gabungan (`combined_features`) dibuat dari:

* Category
* Skin Type
* Usage frequency
* Ingredients

Kemudian dilakukan:

```
TF-IDF → Cosine Similarity
```

Sehingga produk yang mirip dari sisi konten dapat dikenali.

## 2️⃣ Hybrid Scoring

Produk diberi skor gabungan:

```
Weighted Score = (α × Rating) + (β × Popularity) + (γ × Similarity)
```

Default:

* α = 0.4
* β = 0.2
* γ = 0.4

## 3️⃣ Evaluasi Model

Sistem dievaluasi menggunakan:

* Precision@K
* Recall@K
* NDCG@K

Evaluasi dilakukan pada **K = 20**, dan hasilnya sebagai berikut:

| Metrik       | Nilai  |
| ------------ | ------ |
| Precision@20 | 0.75   |
| Recall@20    | 0.75   |
| NDCG@20      | 0.8623 |

Dari hasil ini dapat disimpulkan bahwa:

* Model mampu memberikan rekomendasi yang relevan dengan tingkat presisi 75%.
* Recall 75% menunjukkan bahwa sebagian besar item relevan berhasil direkomendasikan.
* Nilai NDCG yang tinggi (0.8623) menandakan bahwa sistem tidak hanya memberikan item yang benar, namun juga menempatkannya dalam urutan yang tepat.

---

# 📁 Tentang Dataset

Dataset ini adalah **data sintetis (simulated)** mengenai produk kecantikan di seluruh dunia, digunakan untuk:

* Data science
* Analisis kebiasaan konsumen
* Sistem rekomendasi
* Pembelajaran machine learning

### 🧴 Kategori Produk

Termasuk:

* Skincare
* Makeup
* Haircare
* Fragrance
* Personal Care

### 📊 Kolom Dataset Utama

| Kolom             | Deskripsi           |
| ----------------- | ------------------- |
| Product_Name      | Nama produk         |
| Brand             | Merek               |
| Category          | Kategori            |
| Price             | Harga USD           |
| Rating            | Skor konsumen       |
| Number_of_Reviews | Jumlah review       |
| Skin_Type         | Jenis kulit         |
| Gender            | Target pengguna     |
| Price_IDR         | Harga Rupiah        |
| combined_features | Gabungan fitur teks |
| Image_URL         | URL gambar          |

⚠ Dataset ini **100% tidak berasal dari data asli**, hanya untuk penelitian dan pembelajaran.

---

# 📂 Struktur Project

```
.
├── app.py                            → Aplikasi Streamlit
├── requirements.txt                  → Dependency
├── modules/
│   ├── user_filter.py                → Filter preferensi user
│   ├── recommendation.py             → Hybrid ranking
│   ├── image.py                      → Generator gambar dummy
│   └── data_preprocessing.py         → Data cleaning
├── precompute_similarity.py          → Build TF-IDF & cosine matrix
├── data/
│   ├── skincare_products.csv
│   ├── skincare_products_clean.csv
│   └── images/
└── README.md
```

---

# 🧹 Data Preprocessing

Preprocessing mencakup:

✔ Menghapus data tidak valid
✔ Standarisasi nama kolom
✔ Melengkapi nilai hilang
✔ Konversi harga USD → IDR
✔ Lowercase normalization
✔ Membuat `combined_features` sebagai input TF-IDF
✔ Menambahkan `Image_URL`

---

# 🚀 Cara Menjalankan Aplikasi

## 1️⃣ Install dependency

```
pip install -r requirements.txt
```

## 2️⃣ Jalankan preprocessing (jika diperlukan)

```
python data_preprocessing.py
```

## 3️⃣ Jalankan aplikasi Streamlit

```
streamlit run app.py
```

---

# 🖥 Cara Menggunakan

1. Jalankan aplikasi
2. Pilih preferensi pada **sidebar**
3. Sistem akan:

   * Memfilter produk
   * Menghitung kesamaan
   * Melakukan ranking hybrid

---

# 🌐 Link Deploy Streamlit

Jika sudah dideploy ke Streamlit Cloud, link akan berbentuk:

```
https://<username>-skincare-recommender.streamlit.app
```

Tambahkan link setelah aplikasi online.

---

# 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik dan bebas dimodifikasi selama mencantumkan kredit.
