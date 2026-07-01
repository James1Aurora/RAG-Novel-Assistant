# Requirements Document

## Introduction

RAG Novel Assistant adalah aplikasi berbasis Retrieval-Augmented Generation (RAG) yang memungkinkan penulis novel dan pembaca untuk mengupload dokumen novel dalam format PDF, TXT, atau DOCX, lalu mengajukan pertanyaan tentang isi novel tersebut — seperti karakter, alur cerita, latar, konflik, dan tema. Aplikasi ini menggunakan LLM lokal melalui Ollama (llama3/mistral) dan embedding nomic-embed-text, sehingga sepenuhnya berjalan secara lokal tanpa biaya API eksternal. Jawaban selalu didasarkan pada konten dokumen yang diupload, bukan pengetahuan umum LLM.

## Glossary

- **RAG_System**: Sistem keseluruhan RAG Novel Assistant
- **Document_Loader**: Komponen yang membaca dan memuat file dokumen (PDF, TXT, DOCX)
- **Chunker**: Komponen yang memecah dokumen menjadi potongan-potongan teks (chunk)
- **Embedder**: Komponen yang mengubah teks chunk menjadi vektor numerik menggunakan model nomic-embed-text
- **Vector_Store**: Komponen ChromaDB yang menyimpan dan mengindeks vektor embedding
- **Retriever**: Komponen yang mengambil chunk paling relevan berdasarkan query pengguna
- **LLM**: Model bahasa besar yang berjalan lokal via Ollama (llama3 atau mistral)
- **RAG_Pipeline**: Alur kerja end-to-end dari query pengguna hingga jawaban akhir
- **UI**: Antarmuka pengguna berbasis Streamlit
- **Chunk**: Potongan teks dari dokumen dengan metadata (nama file, posisi)
- **Metadata**: Informasi tambahan pada setiap chunk, mencakup nama file dan nomor halaman/bab
- **Novel**: Dokumen fiksi yang diupload pengguna (bahasa Indonesia atau Inggris)
- **Token**: Kata yang dipisahkan oleh spasi (whitespace-delimited word) untuk keperluan chunking

---

## Requirements

### Requirement 1: Upload dan Pemrosesan Dokumen Novel

**User Story:** Sebagai penulis novel, saya ingin mengupload file novel saya dalam berbagai format, sehingga saya dapat mengajukan pertanyaan tentang isi novel tersebut.

#### Acceptance Criteria

1. THE Document_Loader SHALL menerima file dalam format PDF, TXT, dan DOCX melalui UI dengan ukuran maksimal 50 MB per file.
2. WHEN pengguna mengupload file dokumen, THE Document_Loader SHALL memvalidasi bahwa format file adalah PDF, TXT, atau DOCX, dan ukuran file tidak melebihi 50 MB.
3. IF format file tidak didukung (bukan PDF, TXT, atau DOCX), THEN THE UI SHALL menampilkan indikasi error yang menyebutkan format file yang diterima.
4. IF ukuran file melebihi 50 MB, THEN THE UI SHALL menampilkan indikasi error yang menyebutkan batas ukuran file yang diizinkan.
5. WHEN dokumen berhasil dimuat, THE Document_Loader SHALL mengekstrak teks beserta metadata halaman atau bab jika tersedia.
6. WHEN teks berhasil diekstrak DAN chunk berhasil disimpan ke Vector_Store, THE UI SHALL menampilkan konfirmasi bahwa dokumen siap digunakan.

---

### Requirement 2: Chunking Dokumen

**User Story:** Sebagai pengembang sistem, saya ingin dokumen dipecah menjadi chunk yang optimal, sehingga retrieval menjadi lebih akurat dan relevan.

#### Acceptance Criteria

1. THE Chunker SHALL memecah dokumen menjadi chunk per paragraf atau per 500 token (kata dipisahkan spasi), mana yang lebih dahulu tercapai.
2. THE Chunker SHALL menerapkan overlap sebesar 50 token antar chunk yang berurutan dalam dokumen yang sama, tanpa overlap antar dokumen yang berbeda.
3. THE Chunker SHALL menyertakan metadata pada setiap chunk, mencakup nama file sumber dan nomor halaman atau bab (jika tersedia), atau nilai `"tidak diketahui"` jika metadata tidak dapat dideteksi.
4. WHEN sebuah paragraf melebihi 500 token, THE Chunker SHALL memecah paragraf tersebut pada batas kata terdekat pada atau sebelum token ke-500, dengan tetap menerapkan overlap 50 token.
5. THE Chunker SHALL menghasilkan minimal satu chunk untuk setiap dokumen yang mengandung setidaknya 1 token.
6. IF dokumen tidak mengandung token sama sekali (kosong), THEN THE Chunker SHALL mengembalikan daftar chunk kosong dan THE UI SHALL menampilkan pesan bahwa dokumen tidak mengandung teks yang dapat diproses.

---

### Requirement 3: Embedding dan Penyimpanan Vektor

**User Story:** Sebagai pengembang sistem, saya ingin setiap chunk diubah menjadi vektor dan disimpan secara persisten, sehingga retrieval dapat dilakukan dengan cepat pada sesi berikutnya.

#### Acceptance Criteria

1. THE Embedder SHALL mengubah setiap chunk teks menjadi vektor numerik menggunakan model nomic-embed-text via Ollama.
2. THE Vector_Store SHALL menyimpan vektor embedding beserta metadata chunk ke dalam ChromaDB secara persisten, sehingga data tetap tersedia setelah aplikasi di-restart.
3. WHEN pengguna mengupload dokumen dengan nama file yang sama dengan dokumen yang sudah ada di Vector_Store, THE Vector_Store SHALL melewati proses penyimpanan untuk seluruh dokumen tersebut dan THE UI SHALL menampilkan notifikasi bahwa dokumen sudah ada.
4. IF Ollama tidak dapat dijangkau saat proses embedding, THEN THE RAG_System SHALL menghentikan proses embedding dan menampilkan notifikasi kepada UI bahwa Ollama tidak dapat dijangkau.
5. WHEN proses embedding selesai, THE Vector_Store SHALL mengonfirmasi kepada UI jumlah chunk yang berhasil disimpan dan jumlah chunk yang gagal (jika ada).
6. IF ChromaDB tidak dapat diakses saat penyimpanan, THEN THE RAG_System SHALL menampilkan notifikasi error kepada UI dan membatalkan proses penyimpanan.
7. WHEN sebagian chunk gagal dalam proses embedding, THE RAG_System SHALL melanjutkan pemrosesan chunk yang tersisa dan melaporkan jumlah chunk yang gagal kepada UI di akhir proses.

---

### Requirement 4: Retrieval Chunk yang Relevan

**User Story:** Sebagai penulis novel, saya ingin sistem mengambil bagian novel yang paling relevan dengan pertanyaan saya, sehingga jawaban yang diberikan akurat dan berbasis dokumen.

#### Acceptance Criteria

1. WHEN pengguna mengirimkan query, THE Retriever SHALL mengambil hingga 5 chunk dengan skor kemiripan tertinggi dari Vector_Store.
2. THE Retriever SHALL menggunakan embedding model nomic-embed-text yang sama untuk mengubah query pengguna menjadi vektor sebelum pencarian.
3. THE Retriever SHALL menyertakan metadata (nama file dan nomor urut chunk dalam dokumen) dari setiap chunk yang diambil.
4. IF jumlah chunk yang tersedia di Vector_Store kurang dari 5, THEN THE Retriever SHALL mengembalikan semua chunk yang tersedia tanpa error.
5. IF tidak ada dokumen yang telah diproses di Vector_Store, THEN THE Retriever SHALL mengembalikan daftar chunk kosong.
6. IF THE Retriever mengembalikan daftar chunk kosong karena Vector_Store kosong, THEN THE UI SHALL menampilkan pesan "Belum ada dokumen yang diupload. Silakan upload novel terlebih dahulu."
7. THE Retriever SHALL mengembalikan chunk hasil retrieval dalam urutan berdasarkan skor kemiripan tertinggi ke terendah.

---

### Requirement 5: Generasi Jawaban Berbasis Dokumen

**User Story:** Sebagai penulis novel, saya ingin mendapatkan jawaban yang didasarkan sepenuhnya pada isi novel saya, sehingga saya tidak mendapatkan informasi yang tidak akurat dari pengetahuan umum LLM.

#### Acceptance Criteria

1. WHEN Retriever menghasilkan chunk relevan, THE LLM SHALL menghasilkan jawaban berdasarkan konteks dari chunk-chunk tersebut.
2. THE RAG_Pipeline SHALL menyertakan instruksi eksplisit dalam prompt kepada LLM untuk hanya menjawab berdasarkan konteks dokumen yang diberikan, bukan dari pengetahuan umum.
3. IF konten dari chunk yang diambil tidak mengandung informasi yang cukup untuk menjawab query pengguna, THEN THE LLM SHALL merespons dengan pernyataan bahwa informasi tersebut tidak ditemukan dalam dokumen yang diupload.
4. THE RAG_Pipeline SHALL membangun prompt yang menyertakan seluruh teks chunk sebagai konteks dan query pengguna sebelum mengirim ke LLM.
5. WHEN jawaban berhasil dihasilkan, THE RAG_Pipeline SHALL meneruskan jawaban beserta daftar sumber chunk ke UI dalam satu respons.
6. IF LLM tidak dapat dijangkau via Ollama saat generasi jawaban, THEN THE RAG_Pipeline SHALL menghentikan proses dan THE UI SHALL menampilkan notifikasi bahwa LLM tidak dapat dijangkau.

---

### Requirement 6: Tampilan Sumber Referensi

**User Story:** Sebagai penulis novel, saya ingin melihat dari bagian mana jawaban diambil, sehingga saya dapat memverifikasi konteks jawaban secara langsung di novel saya.

#### Acceptance Criteria

1. WHEN jawaban ditampilkan, THE UI SHALL menampilkan daftar sumber referensi di bawah setiap jawaban.
2. THE UI SHALL menampilkan nama file sumber dan posisi dalam format "Halaman [N]" atau "Bab [N]" untuk setiap chunk yang digunakan sebagai referensi.
3. THE UI SHALL menampilkan antara 1 hingga 5 sumber referensi, sesuai dengan jumlah chunk yang dikembalikan oleh Retriever.
4. IF metadata posisi tidak tersedia untuk sebuah chunk, THEN THE UI SHALL menampilkan nama file saja tanpa posisi untuk chunk tersebut.
5. IF Retriever tidak mengembalikan chunk (daftar kosong), THEN THE UI SHALL menampilkan indikasi bahwa tidak ada sumber referensi yang tersedia untuk jawaban tersebut.

---

### Requirement 7: Antarmuka Pengguna Streamlit

**User Story:** Sebagai penulis novel yang tidak teknis, saya ingin antarmuka yang mudah digunakan dalam Bahasa Indonesia, sehingga saya dapat berinteraksi dengan novel saya tanpa pengetahuan teknis.

#### Acceptance Criteria

1. THE UI SHALL menampilkan sidebar yang berisi komponen upload dokumen dan pilihan model Ollama dengan opsi `llama3` dan `mistral`, dengan `llama3` terpilih sebagai default saat aplikasi pertama kali dimuat.
2. THE UI SHALL menampilkan area chat utama untuk tanya jawab dengan riwayat percakapan dalam sesi aktif.
3. THE UI SHALL merender seluruh teks antarmuka — termasuk label, tombol, placeholder, dan pesan error — dalam Bahasa Indonesia.
4. WHEN pengguna memilih model Ollama di sidebar, THE RAG_Pipeline SHALL menggunakan model yang dipilih untuk semua query berikutnya dalam sesi tersebut.
5. WHEN pengguna mengirimkan pertanyaan, THE UI SHALL menampilkan indikator loading hingga respons dari RAG_Pipeline selesai dirender sepenuhnya di area chat.
6. WHILE sesi Streamlit aktif, THE UI SHALL mempertahankan riwayat percakapan hingga maksimal 50 pasang pertanyaan dan jawaban, di mana pasang terlama dihapus terlebih dahulu jika batas tercapai.
7. WHEN RAG_Pipeline menghasilkan jawaban, THE UI SHALL menampilkan referensi sumber di bawah setiap jawaban, mencakup nama file dan posisi chunk (nomor halaman atau bab jika tersedia).
8. IF pengguna mengirimkan pertanyaan tanpa dokumen yang telah berhasil diproses, THEN THE UI SHALL menonaktifkan input pertanyaan dan menampilkan pesan yang menginstruksikan pengguna untuk mengupload dokumen terlebih dahulu.

---

### Requirement 8: Pemilihan dan Konfigurasi Model Ollama

**User Story:** Sebagai pengembang, saya ingin dapat memilih model LLM yang digunakan, sehingga saya dapat menyesuaikan performa dan kapasitas mesin lokal yang tersedia.

#### Acceptance Criteria

1. THE UI SHALL menyediakan opsi pemilihan minimal dua model (llama3 dan mistral) di sidebar, dengan model yang sedang aktif ditampilkan sebagai terpilih.
2. IF model Ollama yang dipilih tidak tersedia atau belum diunduh, THEN THE RAG_System SHALL menampilkan notifikasi yang menyebutkan nama model dan instruksi untuk mengunduhnya, tanpa memproses query lebih lanjut.
3. THE RAG_System SHALL menggunakan model llama3 sebagai model default dalam sesi tersebut jika pengguna belum memilih model secara eksplisit.
4. IF koneksi ke Ollama gagal saat pengguna mengirim query, THEN THE RAG_System SHALL menampilkan notifikasi kegagalan koneksi tanpa mengubah riwayat chat.
5. WHEN pengguna mengganti model aktif di sidebar, THE RAG_Pipeline SHALL menggunakan model baru untuk semua query berikutnya dalam sesi tersebut, tanpa mempengaruhi riwayat percakapan sebelumnya.

---

### Requirement 9: Pengelolaan Sesi dan Dokumen

**User Story:** Sebagai penulis novel, saya ingin dapat mengelola dokumen yang telah saya upload, sehingga saya dapat beralih antar novel atau membersihkan data lama.

#### Acceptance Criteria

1. WHILE sesi Streamlit aktif, THE UI SHALL menampilkan daftar dokumen yang telah berhasil diproses dan tersimpan di Vector_Store, mencakup nama file dan jumlah chunk per dokumen.
2. WHEN pengguna mengupload dokumen baru, THE RAG_System SHALL memproses dan menambahkan dokumen tersebut ke Vector_Store yang sudah ada tanpa menghapus dokumen lain.
3. THE UI SHALL menyediakan tombol untuk menghapus semua dokumen dari Vector_Store dan mengosongkan riwayat percakapan di layar.
4. WHEN pengguna mengklik tombol reset, THE UI SHALL menampilkan dialog konfirmasi sebelum menghapus data; selama dialog aktif, tidak ada data yang dihapus.
5. WHEN pengguna mengkonfirmasi reset, THE Vector_Store SHALL menghapus semua chunk dan embedding dari ChromaDB.
6. WHEN penghapusan Vector_Store berhasil, THE UI SHALL mereset riwayat percakapan dan memperbarui daftar dokumen menjadi kosong.
7. IF penghapusan ChromaDB gagal, THEN THE RAG_System SHALL menampilkan notifikasi error kepada UI dan mempertahankan state Vector_Store serta riwayat percakapan tanpa perubahan.
