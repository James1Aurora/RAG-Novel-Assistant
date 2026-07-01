"""
RAG Pipeline untuk RAG Novel Assistant.

Modul ini mengorkestrasikan alur kerja lengkap dari pemrosesan dokumen
(load → chunk → embed → store) hingga generasi jawaban (retrieve → prompt → LLM).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

from app.exceptions import (
    DuplicateDocumentError,
    EmptyDocumentError,
    EmptyVectorStoreError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    ValidationError,
)
from app.utils import (
    check_ollama_connection,
    count_tokens,
    truncate_to_token_limit,
    validate_file_format,
    validate_file_size,
)
from app.vector_store import VectorStoreManager


# Ukuran file maks default (dalam bytes)
_UKURAN_MAKS_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024


@dataclass
class SourceReference:
    """Referensi sumber untuk satu chunk yang digunakan dalam jawaban."""
    filename: str
    position: str   # "Halaman N", "Bab N", atau "" jika tidak diketahui
    chunk_index: int


@dataclass
class ProcessResult:
    """Hasil pemrosesan dokumen."""
    success: bool
    chunks_stored: int
    chunks_failed: int
    message: str


@dataclass
class QueryResult:
    """Hasil query RAG."""
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    error: str | None = None


class RAGPipeline:
    """Pipeline RAG end-to-end untuk RAG Novel Assistant.

    Mengorkestrasikan proses upload dokumen (validasi → ekstraksi → chunking →
    embedding → penyimpanan) dan proses query (embed query → retrieval →
    prompt building → generasi jawaban LLM).

    """

    def __init__(self, vector_store: VectorStoreManager, ollama_base_url: str):
        """Menginisialisasi pipeline dengan vector store dan URL Ollama.

        Args:
            vector_store: Instance VectorStoreManager yang telah diinisialisasi.
            ollama_base_url: URL dasar layanan Ollama lokal.
        """
        self.vector_store = vector_store
        self.ollama_base_url = ollama_base_url
        self.chunk_size_tokens = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
        self.chunk_overlap_tokens = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    def load_and_process_document(self, file_path: str, filename: str) -> ProcessResult:
        """Memuat, memvalidasi, men-chunk, dan menyimpan dokumen ke vector store.

        Args:
            file_path: Path lengkap file dokumen di filesystem.
            filename: Nama asli file yang diupload pengguna.

        Returns:
            ProcessResult dengan status, jumlah chunk tersimpan, dan pesan.

        Raises:
            ValidationError: Jika format atau ukuran file tidak valid.
            DuplicateDocumentError: Jika dokumen dengan nama yang sama sudah ada.
            EmptyDocumentError: Jika dokumen tidak mengandung teks.
            OllamaConnectionError: Jika Ollama tidak dapat dijangkau saat embedding.
        """
        # Validasi file
        self._validate_file(file_path, filename)

        # Cek duplikat
        if self.vector_store.document_exists(filename):
            raise DuplicateDocumentError(
                f"Dokumen '{filename}' sudah ada di sistem."
            )

        teks_per_bagian = self._extract_document_text(file_path, filename)
        if not teks_per_bagian:
            raise EmptyDocumentError(
                f"Dokumen '{filename}' tidak mengandung teks yang bisa diproses."
            )

        chunks = self._build_chunks(filename, teks_per_bagian)
        if not chunks:
            raise EmptyDocumentError(
                f"Dokumen '{filename}' tidak menghasilkan chunk yang valid."
            )

        hasil = self.vector_store.store_chunks(chunks)

        return ProcessResult(
            success=hasil.stored > 0,
            chunks_stored=hasil.stored,
            chunks_failed=hasil.failed,
            message=(
                f"Dokumen '{filename}' berhasil diproses. "
                f"{hasil.stored} chunk tersimpan."
            ),
        )

    def query(self, question: str, model: str) -> QueryResult:
        """Menjalankan pipeline RAG lengkap untuk sebuah pertanyaan.

        Args:
            question: Pertanyaan yang diajukan pengguna.
            model: Nama model Ollama yang digunakan untuk generasi jawaban.

        Returns:
            QueryResult dengan jawaban dan daftar sumber referensi.

        Raises:
            EmptyVectorStoreError: Jika belum ada dokumen di vector store.
            OllamaConnectionError: Jika Ollama tidak dapat dijangkau.
            OllamaModelNotFoundError: Jika model yang dipilih belum tersedia.
        """
        # Cek vector store kosong
        if self.vector_store.get_chunk_count() == 0:
            raise EmptyVectorStoreError(
                "Belum ada dokumen yang diupload. Silakan upload novel terlebih dahulu."
            )

        if not check_ollama_connection(self.ollama_base_url):
            raise OllamaConnectionError(
                f"Ollama tidak dapat dijangkau di {self.ollama_base_url}."
            )

        self._ensure_model_available(model)

        chunks = self.vector_store.similarity_search(question, k=5)
        if not chunks:
            return QueryResult(
                answer="Informasi ini tidak ditemukan dalam dokumen yang diupload.",
                sources=[],
            )

        sumber: list[SourceReference] = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {}) or {}
            sumber.append(
                SourceReference(
                    filename=metadata.get("filename", "unknown"),
                    position=metadata.get("position", ""),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                )
            )

        prompt = self._build_prompt(chunks, question)
        jawaban = self._generate_answer(prompt, model)

        return QueryResult(answer=jawaban, sources=sumber)

    def _validate_file(self, file_path: str, filename: str) -> None:
        """Memvalidasi format dan ukuran file.

        Args:
            file_path: Path file di filesystem untuk mengecek ukuran.
            filename: Nama file untuk mengecek format/ekstensi.

        Raises:
            ValidationError: Jika format tidak didukung atau ukuran melebihi batas.
        """
        if not validate_file_format(filename):
            raise ValidationError(
                f"Format file tidak didukung. Gunakan PDF, TXT, atau DOCX. "
                f"(Diterima: '{Path(filename).suffix}')"
            )

        ukuran = os.path.getsize(file_path)
        if not validate_file_size(ukuran):
            ukuran_mb = ukuran / (1024 * 1024)
            raise ValidationError(
                f"Ukuran file melebihi batas 50 MB. "
                f"(Ukuran file: {ukuran_mb:.1f} MB)"
            )

    def _build_prompt(self, chunks: list, question: str) -> str:
        """Membangun prompt dengan instruksi eksplisit untuk menjawab berdasarkan konteks.

        Args:
            chunks: List chunk teks yang relevan sebagai konteks.
            question: Pertanyaan pengguna.

        Returns:
            String prompt yang siap dikirim ke LLM.
        """
        konteks = "\n\n".join(
            chunk if isinstance(chunk, str) else chunk.get("text", "")
            for chunk in chunks
        )

        return (
            "Kamu adalah asisten yang membantu menjawab pertanyaan tentang novel berdasarkan\n"
            "dokumen yang diberikan. PENTING: Jawab HANYA berdasarkan konteks di bawah ini.\n"
            "Jangan gunakan pengetahuan umum yang tidak ada dalam konteks.\n"
            "Jika informasi tidak ditemukan dalam konteks, jawab dengan:\n"
            '"Informasi ini tidak ditemukan dalam dokumen yang diupload."\n\n'
            f"Konteks dari novel:\n{konteks}\n\n"
            f"Pertanyaan: {question}\n\n"
            "Jawaban:"
        )

    def _extract_document_text(self, file_path: str, filename: str) -> list[dict[str, Any]]:
        ekstensi = Path(filename).suffix.lower()

        if ekstensi == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file_handle:
                teks = file_handle.read().strip()
            return [{"text": teks, "position": "", "source_type": "text"}] if teks else []

        if ekstensi == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            hasil: list[dict[str, Any]] = []
            for page_index, page in enumerate(reader.pages, 1):
                teks = (page.extract_text() or "").strip()
                if teks:
                    hasil.append(
                        {
                            "text": teks,
                            "position": f"Halaman {page_index}",
                            "source_type": "pdf",
                        }
                    )
            return hasil

        if ekstensi == ".docx":
            from docx import Document

            document = Document(file_path)
            paragraf = [p.text.strip() for p in document.paragraphs if p.text.strip()]
            teks = "\n\n".join(paragraf).strip()
            return [{"text": teks, "position": "", "source_type": "docx"}] if teks else []

        return []

    def _build_chunks(self, filename: str, bagian_teks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        chunk_index_global = 0

        for bagian in bagian_teks:
            teks = bagian.get("text", "").strip()
            if not teks:
                continue

            potongan = truncate_to_token_limit(
                teks,
                max_tokens=self.chunk_size_tokens,
                overlap=self.chunk_overlap_tokens,
            )

            for local_index, isi_chunk in enumerate(potongan, 1):
                if count_tokens(isi_chunk) == 0:
                    continue

                chunks.append(
                    {
                        "text": isi_chunk,
                        "metadata": {
                            "filename": filename,
                            "chunk_index": chunk_index_global,
                            "position": bagian.get("position", ""),
                            "source_type": bagian.get("source_type", "unknown"),
                            "chunk_number_in_source": local_index,
                        },
                    }
                )
                chunk_index_global += 1

        return chunks

    def _ensure_model_available(self, model: str) -> None:
        import urllib.request

        url = self.ollama_base_url.rstrip("/") + "/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status != 200:
                    raise OllamaConnectionError(
                        f"Ollama tidak merespons dengan benar di {self.ollama_base_url}."
                    )
                data = json.loads(response.read().decode("utf-8"))
                model_names = {
                    item.get("name")
                    for item in data.get("models", [])
                    if item.get("name")
                }
                model_aliases = {
                    nama.split(":", 1)[0]
                    for nama in model_names
                    if nama
                }
                if model not in model_names and model not in model_aliases:
                    raise OllamaModelNotFoundError(
                        f"Model '{model}' belum tersedia. Jalankan: ollama pull {model}"
                    )
        except OllamaModelNotFoundError:
            raise
        except Exception as exc:
            raise OllamaConnectionError(f"Gagal memeriksa model Ollama: {exc}") from exc

    def _generate_answer(self, prompt: str, model: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")

        req = request.Request(
            self.ollama_base_url.rstrip("/") + "/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                jawaban = (data.get("response") or "").strip()
                if not jawaban:
                    return "Informasi ini tidak ditemukan dalam dokumen yang diupload."
                return jawaban
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            if exc.code == 404 or "model" in body.lower():
                raise OllamaModelNotFoundError(
                    f"Model '{model}' belum tersedia. Jalankan: ollama pull {model}"
                ) from exc
            raise OllamaConnectionError(
                f"Gagal menghasilkan jawaban dari Ollama: {body or exc.reason}"
            ) from exc
        except Exception as exc:
            raise OllamaConnectionError(f"Gagal menghubungi Ollama: {exc}") from exc
