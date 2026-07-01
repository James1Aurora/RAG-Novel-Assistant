# Project: RAG Novel Assistant

## Deskripsi Project
Aplikasi RAG (Retrieval-Augmented Generation) untuk membantu penulisan dan eksplorasi novel. Pengguna dapat mengupload dokumen novel (PDF/TXT) dan bertanya jawab tentang isi novel tersebut — karakter, alur, latar, dll.

## Stack Teknologi
- **Runtime**: Python 3.10+
- **LLM**: Ollama (lokal, gratis) — model default: `llama3` atau `mistral`
- **Embedding**: Ollama embedding model (`nomic-embed-text`)
- **RAG Framework**: LangChain
- **Vector Database**: ChromaDB (lokal, persisten)
- **UI**: Streamlit
- **File yang didukung**: PDF, TXT, DOCX

## Konvensi Kode
- Gunakan **Python** sebagai bahasa utama
- Semua kode ditulis dalam **Bahasa Indonesia** untuk komentar dan docstring
- Struktur folder:
  ```
  /app
    main.py          # entry point Streamlit
    rag_pipeline.py  # RAG pipeline (load, chunk, embed, retrieve)
    vector_store.py  # ChromaDB management
    utils.py         # helper functions
  /docs              # folder upload dokumen novel
  /chroma_db         # folder persistent vector store
  requirements.txt
  .env.example
  ```
- Gunakan `python-dotenv` untuk konfigurasi
- Fungsi harus punya docstring singkat dalam Bahasa Indonesia

## Konteks Domain: Novel
- Dokumen yang diproses adalah **novel fiksi** (bisa bahasa Indonesia atau Inggris)
- Pertanyaan tipikal pengguna:
  - "Siapa karakter utama di bab ini?"
  - "Apa konflik yang terjadi antara X dan Y?"
  - "Di mana latar cerita bagian ini?"
  - "Ringkaskan alur bab 3"
- Chunking strategy: **per paragraf atau per 500 token**, dengan overlap 50 token
- Metadata yang disimpan per chunk: nama file, nomor halaman/bab (jika tersedia)

## Aturan RAG Pipeline
- Retrieval: ambil **top 5 chunk** paling relevan
- Selalu tampilkan **sumber chunk** (nama file + posisi) di bawah jawaban
- Jika jawaban tidak ditemukan di dokumen, LLM harus jawab: *"Informasi ini tidak ditemukan dalam dokumen yang diupload."*
- Jangan biarkan LLM menjawab dari pengetahuan umum jika topiknya spesifik ke isi novel

## Preferensi UI (Streamlit)
- Sidebar untuk upload dokumen dan pilih model Ollama
- Area utama untuk chat (tanya jawab)
- Tampilkan sumber referensi di bawah setiap jawaban
- Bahasa UI: Bahasa Indonesia
