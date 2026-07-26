# 📚 Web App Recommendation - Perpustakaan UNP

> **Live Demo App:** [rekomendasi-buku-unp-web-app.streamlit.app](https://rekomendasi-buku-unp-web-app.streamlit.app/)

Aplikasi web interaktif berbasis Streamlit yang menyajikan sistem rekomendasi peminjaman buku perpustakaan Universitas Negeri Padang (UNP). Aplikasi ini memanfaatkan algoritma **Association Rule Mining (FP-Growth)** untuk menganalisis pola transaksi peminjaman buku mahasiswa.

---

## 📸 Tampilan Aplikasi

| Halaman Utama (Home) | Cari Rekomendasi Buku |
| :---: | :---: |
| *(Unggah screenshot Home)* | *(Unggah screenshot Rekomendasi)* |

---

## 💡 Algoritma & Metodologi

Aplikasi ini mengimplementasikan **Market Basket Analysis** menggunakan algoritma **FP-Growth (Frequent Pattern Growth)** dari pustaka `mlxtend`:

1. **Transaction Encoding:** Mengubah dataset transaksi transaksi peminjaman buku menjadi matriks boolean (*one-hot encoded transaction format*) menggunakan `TransactionEncoder`.
2. **Frequent Itemsets Mining:** Ekstraksi itemset buku yang sering dipinjam secara bersamaan menggunakan algoritma **`fpgrowth`**.
3. **Association Rules Extraction:** Pembentukan aturan asosiasi (*Association Rules*) menggunakan fungsi `association_rules` untuk menghasilkan nilai *antecedents* (buku acuan yang dipinjam) dan *consequents* (buku yang direkomendasikan).
4. **Filtering & Segmentasi:** Menggabungkan aturan asosiasi (*rule merging*) untuk penyaringan berdasarkan judul buku spesifik dan preferensi per **Fakultas**.

---

## 🔄 Alur Proses Aplikasi (*Workflow*)

┌──────────────────────────┐
│  Dataset Transaksi Excel │ (DATA PENELITIAN4.xlsx)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Preprocessing Data &     │ (Pandas, Scipy/Numpy)
│ Mining Aturan Asosiasi   │ (FP-Growth & Association Rules)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  File Hasil Merge Rule   │ (Hasilmerge.xlsx, Hasilmerge2.xlsx, Hasilmerge3.xlsx)
└────────────┬─────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Aplikasi Web (Streamlit)                        │
├───────────────────┬───────────────────────────────┬────────────────────┤
│ 🏠 Home           │ 📊 Eksplorasi Data           │ 🔍 Rekomendasi    │
│ - Profil Video UNP│ - Plot Peminjaman vs Thn Masuk│ - Berdasar Judul   │
│                   │ - Plot Peminjaman vs Fakultas │ - Berdasar Fakultas│
│                   │ - Top 3 Rekomendasi Utama     │                    │
└───────────────────┴───────────────────────────────┴────────────────────┘
