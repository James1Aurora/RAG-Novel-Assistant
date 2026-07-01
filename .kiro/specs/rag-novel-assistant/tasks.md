# Implementation Plan: RAG Novel Assistant

## Overview

Implementasi RAG Novel Assistant menggunakan Python dengan stack: Streamlit (UI), LangChain (RAG framework), ChromaDB (vector store persisten), dan Ollama (LLM + embedding lokal). Rencana ini membangun sistem secara inkremental — dimulai dari fondasi proyek dan utilitas, kemudian pipeline inti (chunking, embedding, retrieval), lalu UI Streamlit, dan diakhiri dengan integrasi penuh dan pengujian.

## Tasks

- [ ] 1. Setup struktur proyek, konfigurasi, dan exception hierarchy
  - Buat struktur folder `/app`, `/docs`, `/chroma_db`
  - Buat file `requirements.txt` dengan dependensi: `streamlit`, `langchain`, `langchain-community`, `chromadb`, `python-dotenv`, `pypdf`, `python-docx`, `hypothesis`, `pytest`
  - Buat file `.env.example` dengan semua variabel konfigurasi sesuai desain
  - Buat file `app/exceptions.py` yang mendefinisikan hierarki exception: `RAGNovelError`, `ValidationError`, `OllamaConnectionError`, `OllamaModelNotFoundError`, `ChromaDBError`, `DuplicateDocumentError`, `EmptyDocumentError`, `EmptyVectorStoreError`
  - _Requirements: 1.3, 1.4, 3.4, 3.6, 5.6, 8.2, 9.7_

- [ ] 2. Implementasi `utils.py` — fungsi helper
  - [ ] 2.1 Implementasi fungsi validasi dan token helper di `app/utils.py`
    - Tulis `validate_file_format(filename: str) -> bool` — cek ekstensi `.pdf`, `.txt`, `.docx`
    - Tulis `validate_file_size(file_size_bytes: int, max_mb: int = 50) -> bool`
    - Tulis `count_tokens(text: str) -> int` — hitung kata dipisahkan spasi
    - Tulis `truncate_to_token_limit(text: str, max_tokens: int, overlap: int = 50) -> list[str]`
    - Tulis `format_source_reference(filename: str, page: int | None, chapter: int | None) -> str`
    - Tulis `check_ollama_connection(base_url: str) -> bool`
    - Semua fungsi dilengkapi docstring Bahasa Indonesia
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 6.2, 6.4_

  - [ ]* 2.2 Tulis property test untuk `validate_file_format` dan `validate_file_size`
    - **Property 1: Validasi format dan ukuran file konsisten**
    - **Validates: Requirements 1.1, 1.2**
    - Buat `app/tests/test_properties.py` dengan `test_file_validation_consistent` menggunakan `@given` Hypothesis
    - Tag komentar: `# Feature: rag-novel-assistant, Property 1`

  - [ ]* 2.3 Tulis property test untuk `format_source_reference`
    - **Property 12: Format referensi sumber konsisten**
    - **Validates: Requirements 6.2, 6.4**
    - Tambahkan `test_source_reference_format_consistent` di `app/tests/test_properties.py`
    - Tag komentar: `# Feature: rag-novel-assistant, Property 12`

  - [ ]* 2.4 Tulis unit test untuk `utils.py`
    - Buat `app/tests/test_utils.py`
    - Test kasus: format valid (pdf/txt/docx), format tidak valid (exe/jpg), ukuran di batas, ukuran melebihi batas
    - Test `count_tokens` dengan string kosong, satu kata, banyak spasi
    - Test `format_source_reference` dengan page=None, chapter=None, keduanya tersedia
    - _Requirements: 1.1, 1.2, 6.2, 6.4_


- [ ] 3. Implementasi `app/rag_pipeline.py` — Document Loader dan Chunker
  - [ ] 3.1 Implementasi data classes dan Document Loader
    - Definisikan dataclass `ProcessResult`, `QueryResult`, `SourceReference` di `app/rag_pipeline.py`
    - Implementasi `RAGPipeline.__init__` dengan parameter `vector_store` dan `ollama_base_url`
    - Implementasi `_validate_file(file_path, filename)` — raise `ValidationError` untuk format/ukuran tidak valid
    - Implementasi `_extract_text(file_path)` menggunakan LangChain loaders: `PyPDFLoader`, `TextLoader`, `Docx2txtLoader`
    - Ekstrak metadata halaman (PDF) dan simpan di Document metadata
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 3.2 Tulis unit test untuk `_validate_file` dan `_extract_text`
    - Buat `app/tests/test_rag_pipeline.py`
    - Test format tidak didukung → `ValidationError`
    - Test ukuran melebihi batas → `ValidationError`
    - Test ekstraksi PDF, TXT, DOCX dengan mock file
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ] 3.3 Implementasi `_chunk_documents` — strategi chunking dengan overlap
    - Implementasi chunking per paragraf atau per 500 token (mana lebih dahulu)
    - Terapkan overlap 50 token antar chunk berurutan dalam dokumen yang sama
    - Setiap chunk menyertakan metadata: `filename`, `page_number`, `chapter`, `chunk_index`, `token_count`, `source_type`
    - Dokumen kosong → kembalikan list kosong
    - Chunk ID format: `"{filename}_{chunk_index}"`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 3.4 Tulis property tests untuk chunking
    - **Property 2: Chunk tidak melebihi 500 token** — `test_chunks_within_token_limit`
    - **Validates: Requirements 2.1, 2.4**
    - **Property 3: Overlap 50 token antar chunk berurutan** — `test_chunk_overlap_consistency`
    - **Validates: Requirements 2.2**
    - **Property 4: Metadata lengkap di setiap chunk** — `test_chunk_metadata_completeness`
    - **Validates: Requirements 2.3**
    - **Property 5: Dokumen non-kosong menghasilkan minimal satu chunk** — `test_non_empty_document_produces_chunks`
    - **Validates: Requirements 2.5**
    - Tambahkan semua ke `app/tests/test_properties.py`
    - Tag komentar: `# Feature: rag-novel-assistant, Property 2/3/4/5`

  - [ ]* 3.5 Tulis unit test untuk `_chunk_documents`
    - Buat `app/tests/test_chunking.py`
    - Test dokumen kosong → list kosong
    - Test satu paragraf pendek → satu chunk
    - Test paragraf melebihi 500 token → dipecah pada batas kata
    - Test overlap: 50 token terakhir chunk-N = 50 token pertama chunk-(N+1)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [ ] 4. Checkpoint — Pastikan semua unit test dan property test utils + chunking lulus
  - Pastikan semua tests lulus, tanyakan kepada pengguna jika ada pertanyaan.


- [ ] 5. Implementasi `app/vector_store.py` — Vector Store Manager
  - [ ] 5.1 Implementasi data classes dan `VectorStoreManager`
    - Definisikan dataclass `DocumentInfo` dan `StorageResult`
    - Implementasi `VectorStoreManager.__init__` — inisialisasi ChromaDB dengan `persist_directory` dan `embedding_function`
    - Implementasi `document_exists(filename)` — cek berdasarkan metadata `filename`
    - Implementasi `get_chunk_count()` — total chunk tersimpan
    - Implementasi `get_document_list()` — kembalikan `list[DocumentInfo]` dengan nama file dan jumlah chunk
    - _Requirements: 3.2, 3.3, 9.1_

  - [ ] 5.2 Implementasi `store_chunks` dan `delete_all`
    - Implementasi `store_chunks(chunks)` — simpan chunk+embedding ke ChromaDB, kembalikan `StorageResult`
    - Strategi partial failure: lanjutkan chunk lain jika satu gagal, catat `stored` dan `failed`
    - Implementasi `delete_all()` — hapus semua data ChromaDB, kembalikan `bool`
    - Raise `ChromaDBError` jika ChromaDB tidak dapat diakses
    - _Requirements: 3.2, 3.5, 3.6, 3.7, 9.5_

  - [ ] 5.3 Implementasi `similarity_search`
    - Implementasi `similarity_search(query_vector, k=5)` — kembalikan top-k chunk terurut descending by score
    - Jika store kosong → kembalikan list kosong
    - Jika N < k → kembalikan semua N chunk yang ada
    - _Requirements: 4.1, 4.4, 4.5, 4.7_

  - [ ]* 5.4 Tulis property tests untuk vector store
    - **Property 6: Deduplikasi dokumen berdasarkan nama file** — `test_document_deduplication`
    - **Validates: Requirements 3.3**
    - **Property 7: Akuntansi penyimpanan chunk akurat** — `test_storage_accounting_accurate`
    - **Validates: Requirements 3.5**
    - **Property 8: Retrieval tidak melebihi top-K** — `test_retrieval_max_k_chunks`
    - **Validates: Requirements 4.1, 4.4, 4.5**
    - **Property 10: Urutan hasil retrieval descending** — `test_retrieval_ordered_by_score`
    - **Validates: Requirements 4.7**
    - **Property 14: Penambahan dokumen baru tidak menghapus dokumen lama** — `test_new_document_preserves_existing`
    - **Validates: Requirements 9.2**
    - **Property 15: Reset mengosongkan semua data** — `test_reset_clears_all_data`
    - **Validates: Requirements 9.5, 9.6**
    - Tambahkan ke `app/tests/test_properties.py`
    - Tag komentar: `# Feature: rag-novel-assistant, Property 6/7/8/10/14/15`

  - [ ]* 5.5 Tulis unit test untuk `vector_store.py`
    - Buat `app/tests/test_vector_store.py` dengan ChromaDB di-mock
    - Test `document_exists` untuk dokumen ada dan tidak ada
    - Test `store_chunks` partial failure — StorageResult akurat
    - Test `delete_all` → `get_chunk_count()` == 0
    - Test `similarity_search` dengan store kosong → list kosong
    - _Requirements: 3.2, 3.3, 3.5, 3.6, 3.7, 4.1, 4.4, 4.5, 9.5_


- [ ] 6. Implementasi `rag_pipeline.py` — Embedding, Prompt Builder, dan Query
  - [ ] 6.1 Implementasi `load_and_process_document`
    - Hubungkan `_validate_file` → `_extract_text` → `_chunk_documents` → `vector_store.document_exists` → `vector_store.store_chunks`
    - Raise `DuplicateDocumentError` jika dokumen sudah ada
    - Raise `OllamaConnectionError` jika Ollama tidak dapat dijangkau saat embedding
    - Kembalikan `ProcessResult` dengan status, `chunks_stored`, `chunks_failed`, dan pesan
    - _Requirements: 1.2, 1.5, 1.6, 3.3, 3.4, 3.5, 3.7_

  - [ ] 6.2 Implementasi `_build_prompt` dan `query`
    - Implementasi `_build_prompt(chunks, question)` menggunakan prompt template dari desain (instruksi eksplisit + fallback Bahasa Indonesia)
    - Implementasi `query(question, model)`:
      - Embed query via Ollama nomic-embed-text
      - `similarity_search` → raise `EmptyVectorStoreError` jika hasil kosong
      - `_build_prompt` → kirim ke Ollama LLM
      - Kembalikan `QueryResult` dengan `answer` dan `list[SourceReference]`
      - Handle `OllamaConnectionError` dan `OllamaModelNotFoundError`
    - _Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 8.2_

  - [ ]* 6.3 Tulis property test untuk `_build_prompt`
    - **Property 9: Metadata lengkap di setiap hasil retrieval** — `test_retrieval_metadata_completeness`
    - **Validates: Requirements 4.3**
    - **Property 11: Prompt mengandung instruksi konteks dan fallback** — `test_prompt_contains_required_instructions`
    - **Validates: Requirements 5.2, 5.4**
    - Tambahkan ke `app/tests/test_properties.py`
    - Tag komentar: `# Feature: rag-novel-assistant, Property 9/11`

  - [ ]* 6.4 Tulis unit test untuk `rag_pipeline.py`
    - Buat / perluas `app/tests/test_rag_pipeline.py` (Ollama dan ChromaDB di-mock)
    - Test dokumen duplikat → `DuplicateDocumentError`
    - Test Ollama tidak bisa dijangkau saat embedding → `OllamaConnectionError`
    - Test vector store kosong saat query → `EmptyVectorStoreError` + pesan yang benar
    - Test model belum diunduh → `OllamaModelNotFoundError` dengan instruksi `ollama pull`
    - Test `_build_prompt` mengandung teks chunk dan instruksi fallback
    - _Requirements: 3.3, 3.4, 5.2, 5.3, 5.4, 5.6, 8.2_

- [ ] 7. Checkpoint — Pastikan semua unit test pipeline dan vector store lulus
  - Pastikan semua tests lulus, tanyakan kepada pengguna jika ada pertanyaan.


- [ ] 8. Implementasi `app/main.py` — UI Streamlit
  - [ ] 8.1 Implementasi session state dan sidebar
    - Inisialisasi `st.session_state`: `chat_history`, `selected_model`, `documents_ready`, `processing`, `show_reset_dialog`
    - Implementasi `render_sidebar()` — upload widget, dropdown model (llama3/mistral, default llama3), tombol reset
    - Semua label, tombol, dan placeholder dalam Bahasa Indonesia
    - _Requirements: 7.1, 7.3, 8.1, 8.3_

  - [ ] 8.2 Implementasi `handle_upload` dan `render_document_list`
    - Implementasi `handle_upload(uploaded_file, pipeline)` — simpan file sementara, panggil `pipeline.load_and_process_document`, tampilkan konfirmasi atau pesan error yang sesuai
    - Handle `DuplicateDocumentError`, `ValidationError`, `OllamaConnectionError`, `EmptyDocumentError` dengan pesan UI Bahasa Indonesia
    - Implementasi `render_document_list(vector_store)` — tampilkan nama file dan jumlah chunk per dokumen
    - _Requirements: 1.3, 1.4, 1.6, 3.3, 3.4, 3.5, 3.6, 3.7, 9.1_

  - [ ] 8.3 Implementasi `render_chat_area` dan `handle_query`
    - Implementasi `render_chat_area(session_state)` — tampilkan riwayat chat (maks 50 pasang)
    - Implementasi `handle_query(question, model, pipeline)` — tampilkan loading indicator, panggil `pipeline.query`, tambahkan ke `chat_history`, tampilkan jawaban + sumber referensi
    - Input pertanyaan dinonaktifkan jika `documents_ready == False` dengan pesan instruksi upload
    - Limit chat history: hapus pasangan terlama jika > 50 (FIFO)
    - _Requirements: 4.6, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5, 7.2, 7.5, 7.6, 7.7, 7.8, 8.4_

  - [ ] 8.4 Implementasi `handle_reset` dengan dialog konfirmasi
    - Implementasi `handle_reset(vector_store)` — tampilkan `st.dialog` konfirmasi sebelum menghapus
    - Selama dialog aktif: tidak ada data yang dihapus
    - Saat konfirmasi: panggil `vector_store.delete_all()`, reset `chat_history` dan `documents_ready`
    - Handle `ChromaDBError` — tampilkan error, pertahankan state saat ini
    - _Requirements: 9.3, 9.4, 9.5, 9.6, 9.7_

  - [ ] 8.5 Implementasi model switching dan integrasi penuh di `main.py`
    - Saat pengguna mengganti model di sidebar, update `session_state.selected_model`
    - Model baru digunakan untuk semua query berikutnya tanpa mengubah riwayat chat
    - Pastikan `RAGPipeline` dan `VectorStoreManager` diinisialisasi dari `.env`
    - Konfigurasi `OLLAMA_BASE_URL`, `CHROMA_PERSIST_DIR`, `TOP_K_CHUNKS`, dll. dibaca dari environment variables
    - _Requirements: 7.4, 8.3, 8.4, 8.5_

  - [ ]* 8.6 Tulis property test untuk batas riwayat chat
    - **Property 13: Batas riwayat chat tidak terlampaui** — `test_chat_history_max_limit`
    - **Validates: Requirements 7.6**
    - Tambahkan ke `app/tests/test_properties.py`
    - Tag komentar: `# Feature: rag-novel-assistant, Property 13`

  - [ ]* 8.7 Tulis unit test untuk logika UI
    - Buat `app/tests/test_validation.py`
    - Test chat history FIFO: setelah > 50 pasang, pasang terlama terhapus
    - Test input dinonaktifkan saat `documents_ready == False`
    - Test sumber referensi ditampilkan di bawah jawaban
    - _Requirements: 6.1, 6.3, 6.5, 7.6, 7.7, 7.8_


- [ ] 9. Wiring dan integrasi akhir
  - [ ] 9.1 Verifikasi dependency graph komponen
    - Pastikan `main.py` hanya mengimport dari `rag_pipeline.py` dan `vector_store.py`
    - Pastikan `rag_pipeline.py` mengimport dari `vector_store.py` dan `utils.py`
    - Pastikan `vector_store.py` mengimport dari `utils.py`
    - Pastikan tidak ada circular imports
    - _Requirements: 1.1–9.7 (integrasi umum)_

  - [ ] 9.2 Buat `app/tests/conftest.py` dan fixtures bersama
    - Definisikan fixtures pytest: `sample_chunks`, `mock_vector_store`, `mock_pipeline`, `temp_docs_dir`
    - Pastikan semua test file menggunakan fixtures dari conftest untuk konsistensi
    - _Requirements: semua requirement (infrastruktur test)_

  - [ ]* 9.3 Tulis integration tests (opsional, memerlukan Ollama + ChromaDB aktif)
    - Buat `app/tests/integration/test_full_pipeline.py`
    - Test upload PDF nyata → chunking → embedding → query → jawaban dengan sumber
    - Test persistensi ChromaDB: restart app, dokumen masih ada
    - Test koneksi Ollama gagal → error ditangani tanpa crash
    - _Requirements: 3.1, 3.2, 5.1, 8.4_

- [ ] 10. Final checkpoint — Pastikan semua tests lulus dan aplikasi berjalan
  - Jalankan `pytest app/tests/ -v --ignore=app/tests/integration` — semua harus hijau
  - Pastikan `python -m streamlit run app/main.py` dapat dijalankan tanpa error import
  - Tanyakan kepada pengguna jika ada pertanyaan sebelum dianggap selesai.


## Notes

- Tasks bertanda `*` bersifat opsional dan dapat dilewati untuk MVP yang lebih cepat
- Setiap task mereferensikan requirement spesifik untuk keterlacakan
- Checkpoint memastikan validasi inkremental sebelum melanjutkan ke fase berikutnya
- Property tests menggunakan Hypothesis (min 100 iterasi per property)
- Unit tests menggunakan pytest dengan semua dependency eksternal di-mock
- Integration tests memerlukan Ollama dan ChromaDB aktif secara lokal
- Semua komentar dan docstring ditulis dalam Bahasa Indonesia sesuai konvensi proyek
- Konfigurasi dibaca dari `.env` menggunakan `python-dotenv`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3"] },
    { "id": 4, "tasks": ["3.4", "3.5", "5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3"] },
    { "id": 6, "tasks": ["5.4", "5.5", "6.1"] },
    { "id": 7, "tasks": ["6.2"] },
    { "id": 8, "tasks": ["6.3", "6.4", "8.1"] },
    { "id": 9, "tasks": ["8.2", "8.3"] },
    { "id": 10, "tasks": ["8.4", "8.5"] },
    { "id": 11, "tasks": ["8.6", "8.7", "9.1", "9.2"] },
    { "id": 12, "tasks": ["9.3"] }
  ]
}
```
