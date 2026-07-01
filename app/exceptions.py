"""
Hierarki exception untuk RAG Novel Assistant.

Semua exception yang digunakan di seluruh aplikasi didefinisikan di sini
agar penanganan error konsisten dan mudah dilacak.
"""


class RAGNovelError(Exception):
    """Base exception untuk RAG Novel Assistant.

    Semua exception khusus aplikasi mewarisi dari kelas ini sehingga
    dapat ditangkap secara kolektif maupun individual.
    """


class ValidationError(RAGNovelError):
    """File tidak valid: format tidak didukung atau ukuran melebihi batas.

    Diangkat ketika pengguna mengupload file dengan ekstensi selain
    PDF, TXT, atau DOCX, atau ukurannya melebihi batas maksimal yang
    dikonfigurasi (default 50 MB).
    """


class OllamaConnectionError(RAGNovelError):
    """Ollama tidak dapat dijangkau di URL yang dikonfigurasi.

    Diangkat ketika aplikasi gagal terhubung ke layanan Ollama, baik
    saat proses embedding chunk maupun saat generasi jawaban.
    """


class OllamaModelNotFoundError(RAGNovelError):
    """Model Ollama yang diminta belum diunduh atau tidak tersedia.

    Diangkat ketika model yang dipilih pengguna belum tersedia di
    instalasi Ollama lokal. Pesan error harus menyertakan instruksi
    untuk mengunduh model dengan perintah `ollama pull <nama_model>`.
    """


class ChromaDBError(RAGNovelError):
    """ChromaDB tidak dapat diakses atau operasi gagal.

    Diangkat ketika terjadi kegagalan pada operasi ChromaDB seperti
    penyimpanan chunk, pencarian vektor, atau penghapusan data.
    """


class DuplicateDocumentError(RAGNovelError):
    """Dokumen dengan nama file yang sama sudah ada di Vector Store.

    Diangkat ketika pengguna mencoba mengupload dokumen dengan nama file
    yang identik dengan dokumen yang sudah tersimpan di Vector Store.
    Proses penyimpanan dibatalkan untuk mencegah duplikasi data.
    """


class EmptyDocumentError(RAGNovelError):
    """Dokumen tidak mengandung teks yang dapat diproses.

    Diangkat ketika dokumen yang diupload tidak menghasilkan token apapun
    setelah proses ekstraksi teks, misalnya file PDF yang hanya berisi
    gambar tanpa teks yang dapat dibaca.
    """


class EmptyVectorStoreError(RAGNovelError):
    """Vector Store kosong, tidak ada dokumen yang telah diproses.

    Diangkat ketika pengguna mengirimkan query namun belum ada dokumen
    yang tersimpan di Vector Store. UI harus menginstruksikan pengguna
    untuk mengupload dokumen novel terlebih dahulu.
    """
