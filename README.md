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

## 🔄 Alur Proses Aplikasi (*Workflow*)

### 1. 📥 DATA COLLECTION
* **Metadata Judul Buku:** `JUDUL BUKU.xlsx`
* **Riwayat Transaksi Peminjaman:** `DATA PENELITIAN4.xlsx`
  * *Variabel:* Transaksi, Judul Buku, Tahun Masuk, Fakultas, Hari

---

### 2. 🧹 DATA PREPROCESSING & CLEANING
* **Cleansing Data:** Penanganan *missing values* & penyesuaian tipe data (`Tahun_Masuk`).
* **Matriks Transaksi:** Mengelompokkan buku berdasarkan ID transaksi/pustakawan.
* **One-Hot Encoding:** Mengubah format menjadi matriks biner menggunakan `TransactionEncoder` dari `mlxtend`.

---

### 3. 🤖 MACHINE LEARNING MODELING (FP-GROWTH & ASSOCIATION RULES)
* **Algoritma Utama:** FP-Growth (*Frequent Pattern Growth*).
* **Frequent Itemsets Mining:** Menemukan kombinasi buku yang sering dipinjam bersamaan.
* **Association Rules Generation:** Mengekstrak hubungan *Antecedents* ➔ *Consequents*.
* **Evaluasi Metrik:**
  * *Support*
  * *Confidence*
  * *Lift Ratio* ($> 1$)

---

### 4. 🚀 DEPLOYMENT & FEATURE IMPLEMENTATION (STREAMLIT)
* **Penyimpanan Aturan:** `Hasilmerge.xlsx`, `Hasilmerge2.xlsx`, `Hasilmerge3.xlsx`.
* **Navigasi Interaktif:** Menggunakan `streamlit_option_menu`.
* **Fitur Utama:**
  1. **Eksplorasi Data:** Distribusi peminjaman per Fakultas & Tahun Masuk.
  2. **Top Data:** Buku populer berdasarkan Fakultas.
  3. **E-Commerce Style Search:** Rekomendasi berdasarkan input judul buku.

---

### 5. 🔮 SYSTEM IMPROVEMENT & FUTURE ROADMAP
* **Peningkatan Layanan:** Reorganisasi tata letak fisik rak buku perpustakaan.
* **Pengembangan Lanjutan:** Integrasi OPAC *real-time* via API & implementasi *Hybrid Model* (TF-IDF + FP-Growth).
