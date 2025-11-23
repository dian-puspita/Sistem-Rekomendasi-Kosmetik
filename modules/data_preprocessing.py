import pandas as pd

"""
    Preprocessing dataset kosmetik dan menyimpannya ke file baru.

    Steps:
    - Hapus baris tanpa Product_Name atau Brand
    - Rename kolom agar konsisten
    - Isi missing value teks dan numerik
    - Konversi Price USD ke IDR
    - Standarisasi teks (lowercase & strip)
    - Buat fitur gabungan untuk sistem rekomendasi
    - Tambahkan kolom Image_URL kosong
"""

USD_TO_IDR = 15000  # kurs USD ke IDR

def preprocess_and_save(csv_input, csv_output):
    df = pd.read_csv(csv_input)

    # Hapus baris jika Product_Name atau Brand kosong
    df.dropna(subset=['Product_Name','Brand'], inplace=True)

    # Rename kolom agar konsisten
    rename_dict = {}
    if 'Price_USD' in df.columns:
        rename_dict['Price_USD'] = 'Price'
    if 'Gender_Target' in df.columns:
        rename_dict['Gender_Target'] = 'Gender'
    if 'Main_Ingredient' in df.columns:
        rename_dict['Main_Ingredient'] = 'Ingredients'
    df.rename(columns=rename_dict, inplace=True)

    # Isi missing value teks opsional
    text_cols = ['Category','Skin_Type','Gender','Usage_Frequency','Ingredients']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').str.lower().str.strip()

    # Isi missing value numerik
    numeric_cols = ['Price','Rating','Number_of_Reviews']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Konversi Price USD ke IDR
    if 'Price' in df.columns:
        df['Price_IDR'] = df['Price'] * USD_TO_IDR

    # Standarisasi kolom penting
    df['Product_Name'] = df['Product_Name'].str.lower().str.strip()
    df['Brand'] = df['Brand'].str.lower().str.strip()

    # Buat fitur gabungan
    df['combined_features'] = (
        df['Category'].fillna('') + ' ' +
        df['Skin_Type'].fillna('') + ' ' +
        df['Usage_Frequency'].fillna('') + ' ' +
        df['Ingredients'].fillna('')
    )

    # Kolom kosong untuk URL gambar
    df['Image_URL'] = ''

    df.to_csv(csv_output, index=False)
    print(f"Data clean tersimpan di: {csv_output}")

if __name__ == "__main__":
    preprocess_and_save('data/skincare_products.csv', 'data/skincare_products_clean.csv')
