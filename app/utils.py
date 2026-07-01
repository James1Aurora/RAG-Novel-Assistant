"""
Fungsi-fungsi helper untuk RAG Novel Assistant.

Modul ini menyediakan utilitas umum yang digunakan oleh berbagai komponen
aplikasi, termasuk validasi file, penghitungan token, dan pemformatan output.
"""

import os
from pathlib import Path


# Format file yang didukung
FORMAT_YANG_DIDUKUNG = {".pdf", ".txt", ".docx"}

# Batas ukuran file default (50 MB)
UKURAN_MAKS_DEFAULT_MB = 50


def validate_file_format(filename: str) -> bool:
    """Memeriksa apakah ekstensi file adalah PDF, TXT, atau DOCX.

    Args:
        filename: Nama file yang akan diperiksa.

    Returns:
        True jika ekstensi file didukung, False jika tidak.
    """
    ekstensi = Path(filename).suffix.lower()
    return ekstensi in FORMAT_YANG_DIDUKUNG


def validate_file_size(file_size_bytes: int, max_mb: int = UKURAN_MAKS_DEFAULT_MB) -> bool:
    """Memeriksa apakah ukuran file tidak melebihi batas maksimal.

    Args:
        file_size_bytes: Ukuran file dalam bytes.
        max_mb: Batas ukuran maksimal dalam megabytes (default 50 MB).

    Returns:
        True jika ukuran file masih dalam batas, False jika melebihi batas.
    """
    ukuran_maks_bytes = max_mb * 1024 * 1024
    return file_size_bytes <= ukuran_maks_bytes


def count_tokens(text: str) -> int:
    """Menghitung jumlah token (kata dipisahkan spasi) dalam teks.

    Args:
        text: Teks yang akan dihitung tokennya.

    Returns:
        Jumlah token (kata) dalam teks.
    """
    if not text or not text.strip():
        return 0
    return len(text.split())


def truncate_to_token_limit(
    text: str, max_tokens: int, overlap: int = 50
) -> list[str]:
    """Memecah teks menjadi segmen dengan batas token dan overlap.

    Setiap segmen memiliki maksimal max_tokens token. Segmen berikutnya
    dimulai dengan overlap token terakhir dari segmen sebelumnya.

    Args:
        text: Teks yang akan dipecah.
        max_tokens: Jumlah token maksimal per segmen.
        overlap: Jumlah token yang tumpang tindih antar segmen.

    Returns:
        List string, masing-masing berisi maksimal max_tokens token.
    """
    if not text or not text.strip():
        return []

    kata = text.split()
    if len(kata) == 0:
        return []

    segmen = []
    posisi = 0

    while posisi < len(kata):
        akhir = min(posisi + max_tokens, len(kata))
        segmen.append(" ".join(kata[posisi:akhir]))
        if akhir == len(kata):
            break
        posisi = akhir - overlap
        if posisi <= 0:
            posisi = akhir  # hindari infinite loop

    return segmen


def format_source_reference(
    filename: str,
    page: int | None = None,
    chapter: int | str | None = None,
) -> str:
    """Memformat referensi sumber menjadi string yang mudah dibaca.

    Args:
        filename: Nama file sumber dokumen.
        page: Nomor halaman (opsional).
        chapter: Nomor atau nama bab (opsional).

    Returns:
        String referensi yang sudah diformat, contoh: "novel.pdf — Halaman 12"
        atau "novel.pdf — Bab 3".
    """
    referensi = filename

    if page is not None:
        referensi += f" — Halaman {page}"
    elif chapter is not None:
        referensi += f" — Bab {chapter}"

    return referensi


def check_ollama_connection(base_url: str) -> bool:
    """Memeriksa apakah Ollama dapat dijangkau di URL yang diberikan.

    Melakukan permintaan HTTP GET ke endpoint /api/tags Ollama untuk
    memverifikasi bahwa layanan sedang berjalan dan dapat diakses.

    Args:
        base_url: URL dasar layanan Ollama, contoh: "http://localhost:11434".

    Returns:
        True jika Ollama dapat dijangkau, False jika tidak.
    """
    try:
        import urllib.request
        url = base_url.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False
