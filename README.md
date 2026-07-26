# Rekomendasi-Buku-UNP
# 📚 Website Aplikasi Rekomendasi Buku Perpustakaan UNP

> **Live Demo:** [rekomendasi-buku-unp-web-app.streamlit.app](https://rekomendasi-buku-unp-web-app.streamlit.app/)

Aplikasi web berbasis data untuk memberikan rekomendasi buku secara personal interaktif kepada mahasiswa berdasarkan judul buku pilihan maupun preferensi fakultas.

---

## 📸 Tampilan Aplikasi

![Demo Aplikasi](path/to/demo-or-screenshot.gif)
*Sertakan GIF singkat (10-15 detik) atau screenshot aplikasi di sini.*

---

## ✨ Fitur Utama

- **📊 Eksplorasi Data:** Visualisasi interaktif mengenai distribusi buku dan tren peminjaman.
- **🔍 Cari Rekomendasi:** Sistem rekomendasi berbasis judul (*Content-Based Filtering*).
- **🎓 Rekomendasi Berdasarkan Fakultas:** Filter preferensi buku yang disesuaikan dengan fakultas pengguna.

---

## 🛠️ Tech Stack & Library

- **Bahasa Pemrograman:** Python
- **Framework Web:** Streamlit
- **Pembersihan & Manipulasi Data:** Pandas, NumPy
- **Machine Learning / Algoritma:** Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity)
- **Visualisasi Data:** Plotly / Seaborn

---

## 💡 Metodologi & Alur Kerja

1. **Data Preprocessing:** Pembersihan data transaksi peminjaman dan metadata buku perpustakaan.
2. **Feature Extraction:** Ekstraksi fitur teks menggunakan *TF-IDF*.
3. **Similarity Score:** Menghitung kemiripan antar-buku menggunakan *Cosine Similarity*.
4. **Deployment:** Men-deploy aplikasi ke Streamlit Community Cloud.

---

## 🚀 Cara Menjalankan di Komputer Lokal

### 1. Clone Repositori

git clone [https://github.com/username-anda/nama-repositori.git](https://github.com/username-anda/nama-repositori.git)
cd nama-repositori
