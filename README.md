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

▶️1. DATA COLLECTION                                                                     
 • Metadata Judul Buku Perpustakaan (JUDUL BUKU.xlsx)
   DATA COLLECTION                                                                     
 • Metadata Judul Buku Perpustakaan (JUDUL BUKU.xlsx)
 • Riwayat Historis Transaksi Peminjaman (DATA PENELITIAN4.xlsx) (Transaksi, Judul Buku, Tahun Masuk, Fakultas,       Hari)
▶️2. DATA PREPROCESSING & CLEANING                                                       
 • Cleansing data (penanganan missing values & casting tipe data Tahun_Masuk)       
 • Matriks Transaksi: Mengelompokkan buku per transaksi/pustakawan                  
 • One-Hot Encoding menggunakan `TransactionEncoder` dari mlxtend
▶️3. MACHINE LEARNING MODELING (FP-GROWTH & ASSOCIATION RULES)                           
 • Algoritma: FP-Growth (Frequent Pattern Growth)                                    
 • Penambangan Frequent Itemsets (Kombinasi buku yang sering dipinjam bersama)       
 • Generasi Association Rules (Mengekstrak relasi Antecedents ➔ Consequents)          
 • Evaluasi Metrik: Support, Confidence, dan Lift Ratio (> 1)
▶️4. DEPLOYMENT & FEATURE IMPLEMENTATION (STREAMLIT)                                     
 • Merging & Storage Rules (Hasilmerge.xlsx, Hasilmerge2.xlsx, Hasilmerge3.xlsx)     
 • Interaktif Navigation via `streamlit_option_menu`
 • Feature 1: Eksplorasi Data (Distribusi Peminjaman per Fakultas & Tahun Masuk)     
 • Feature 2: Top Data Based on Peminjaman by Fakultas
 • Feature 3: E-Commerce Style Search (Rekomendasi Berdasar Judul Buku)
▶️5. SYSTEM IMPROVEMENT & FUTURE ROADMAP                                                 
 • Peningkatan Layanan: Reorganisasi tata letak fisik rak buku perpustakaan          
 • Integrasi OPAC Real-time via API & Implementasi Hybrid Model (TF-IDF + FP-Growth)
