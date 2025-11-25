# modules/output_visualization.py
from PIL import Image, ImageDraw, ImageFont
import os

"""
    Generate dummy image untuk produk yang tidak memiliki gambar asli.

    Proses:
    - Mengecek apakah folder penyimpanan gambar sudah ada
    - Membuat nama file aman berdasarkan nama produk
    - Jika file belum ada, sistem membuat gambar baru
    - Gambar menggunakan background gradasi sederhana
    - Nama produk ditampilkan pada gambar (maks. 15 karakter)
    - Gambar disimpan pada direktori target dan mengembalikan path file

    Parameter:
    - product_name : nama produk
    - image_dir    : folder penyimpanan hasil gambar
    - size         : ukuran gambar (default 200x200)

    Output:
    - Mengembalikan path file gambar yang dihasilkan
"""

def generate_dummy_image(product_name, image_dir='data/images', size=(200, 200)):

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # Nama file unik berdasarkan product_name
    safe_name = "".join(c if c.isalnum() else "_" for c in product_name.lower())
    filename = os.path.join(image_dir, f"{safe_name}.png")
    
    if not os.path.exists(filename):
        # Buat image
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)

        # Gradient sederhana (vertikal)
        for y in range(size[1]):
            r = int(200 + (50 * y / size[1]))   # abu ke biru
            g = int(200 + (50 * y / size[1]))
            b = int(200 + (50 * y / size[1]))
            draw.line([(0, y), (size[0], y)], fill=(r, g, b))

        # Text: nama produk, truncate max 15 karakter
        text = product_name[:15] + '...' if len(product_name) > 15 else product_name
        try:
            font = ImageFont.load_default()
            w, h = draw.textbbox((0, 0), text, font=font)[2:]  # ukuran text
            draw.text(((size[0]-w)/2, (size[1]-h)/2), text, fill=(0,0,0), font=font)
        except:
            draw.text((10, 10), text, fill=(0, 0, 0))
        
        # Simpan file
        img.save(filename)
    
    return filename
