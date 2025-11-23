import pandas as pd

"""
    Filter dataset berdasarkan preferensi user.
    
    Semua parameter opsional:
    - category: exact match (case-insensitive)
    - skin_type, gender, usage_frequency: substring match (case-insensitive)
    - price_range: tuple (min, max) selalu diterapkan
    
    Returns:
        DataFrame yang sudah difilter
"""

def filter_user_preferences(df, category, skin_type, gender, usage_frequency, price_range):
    if df is None:
        return pd.DataFrame()
    print(f"\n=== DEBUG FILTER USER ===")
    print(f"Input Filter -> Skin: {skin_type}, Gender: {gender}, Usage: {usage_frequency}, Price Range: {price_range}")

    # Bekerja pada salinan df supaya tidak merubah df asli
    filtered = df.copy()

    # Normalisasi kolom teks ONCE (safe even jika sudah normal)
    for col in ['Category', 'Skin_Type', 'Gender', 'Usage_Frequency']:
        if col in filtered.columns:
            filtered[col] = filtered[col].fillna('').astype(str).str.strip().str.lower()
        else:
            # jika kolom tidak ada, buat kolom kosong (supaya filter tidak error)
            filtered[col] = ''

    # --- Filter kategori (exact match) ---
    if category is not None:
        # ensure param normalized
        cat = str(category).strip().lower()
        filtered = filtered[filtered['Category'] == cat]

     # --- Filter skin_type
    if skin_type is not None:
        stype = str(skin_type).strip().lower()
        filtered = filtered[filtered['Skin_Type'].str.contains(stype, na=False)]

     # --- Filter gender
    if gender is not None:
        g = str(gender).strip().lower()
        filtered = filtered[filtered['Gender'].str.contains(g, na=False)]

    # --- Filter usage_frequency
    if usage_frequency is not None:
        u = str(usage_frequency).strip().lower()
        filtered = filtered[filtered['Usage_Frequency'].str.contains(u, na=False)]

    # harga selalu difilter (asumsikan price_range valid tuple)
    if price_range is not None and len(price_range) == 2:
        min_price, max_price = price_range
        # Pastikan Price_IDR ada dan numeric
        if 'Price_IDR' in filtered.columns:
            filtered = filtered[
                (pd.to_numeric(filtered['Price_IDR'], errors='coerce') >= float(min_price)) &
                (pd.to_numeric(filtered['Price_IDR'], errors='coerce') <= float(max_price))
            ]
        else:
            # jika tidak ada kolom price, return empty
            return pd.DataFrame(columns=filtered.columns)

    return filtered.reset_index(drop=True)
