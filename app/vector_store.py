"""
Manajemen Vector Store menggunakan ChromaDB.

Modul ini mengelola semua interaksi dengan ChromaDB untuk penyimpanan
dan retrieval vektor embedding dokumen novel.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from app.exceptions import ChromaDBError, OllamaConnectionError, OllamaModelNotFoundError


@dataclass
class DocumentInfo:
    """Informasi ringkasan sebuah dokumen yang tersimpan di Vector Store."""
    filename: str
    chunk_count: int


@dataclass
class StorageResult:
    """Hasil operasi penyimpanan chunk ke Vector Store."""
    stored: int
    failed: int


class VectorStoreManager:
    """Mengelola ChromaDB untuk penyimpanan dan retrieval vektor.
    """

    def __init__(self, persist_directory: str, embedding_function=None):
        """Menginisialisasi ChromaDB dengan direktori persisten.

        Args:
            persist_directory: Path direktori penyimpanan persisten ChromaDB.
            embedding_function: Fungsi embedding untuk mengubah teks ke vektor.
        """
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "rag_novel_chunks")

        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._in_memory = False
        except Exception:
            self._client = None
            self._collection = None
            self._in_memory = True
            self._dokumen: dict[str, int] = {}
            self._chunks: list[dict[str, Any]] = []

    def document_exists(self, filename: str) -> bool:
        """Memeriksa apakah dokumen dengan nama file tersebut sudah ada.

        Args:
            filename: Nama file yang akan diperiksa.

        Returns:
            True jika dokumen sudah ada, False jika belum.
        """
        if self._in_memory:
            return filename in self._dokumen

        try:
            hasil = self._collection.get(where={"filename": filename}, limit=1)
            return bool(hasil and hasil.get("ids"))
        except Exception as exc:
            raise ChromaDBError(f"Gagal memeriksa dokumen: {exc}") from exc

    def store_chunks(self, chunks: list) -> StorageResult:
        """Menyimpan chunk beserta embedding ke ChromaDB.

        Args:
            chunks: List chunk dokumen yang akan disimpan.

        Returns:
            StorageResult dengan jumlah chunk yang berhasil dan gagal disimpan.
        """
        if not chunks:
            return StorageResult(stored=0, failed=0)

        if self._in_memory:
            filename = chunks[0].get("metadata", {}).get("filename", "unknown")
            self._dokumen[filename] = len(chunks)
            self._chunks.extend(chunks)
            return StorageResult(stored=len(chunks), failed=0)

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        embeddings: list[list[float]] = []
        stored = 0
        failed = 0

        for index, chunk in enumerate(chunks):
            teks = chunk.get("text", "")
            metadata = chunk.get("metadata", {}) or {}
            if not teks.strip():
                failed += 1
                continue

            try:
                embedding = self._embed_text(teks)
                chunk_id = self._build_chunk_id(metadata, teks, index)
                documents.append(teks)
                metadatas.append(metadata)
                ids.append(chunk_id)
                embeddings.append(embedding)
                stored += 1
            except Exception:
                failed += 1

        if ids:
            try:
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            except Exception as exc:
                raise ChromaDBError(f"Gagal menyimpan chunk ke ChromaDB: {exc}") from exc

        return StorageResult(stored=stored, failed=failed)

    def similarity_search(self, query: str | list[float], k: int = 5) -> list:
        """Mencari k chunk paling relevan berdasarkan similarity vektor.

        Args:
            query: Teks query atau vektor query untuk pencarian similarity.
            k: Jumlah chunk teratas yang dikembalikan (default 5).

        Returns:
            List chunk terurut dari skor kemiripan tertinggi ke terendah.
        """
        if self._in_memory:
            if not self._chunks:
                return []
            return self._chunks[:k]

        try:
            query_embedding = query if isinstance(query, list) else self._embed_text(query)
            hasil = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise ChromaDBError(f"Gagal melakukan similarity search: {exc}") from exc

        ids = (hasil.get("ids") or [[]])[0]
        documents = (hasil.get("documents") or [[]])[0]
        metadatas = (hasil.get("metadatas") or [[]])[0]
        distances = (hasil.get("distances") or [[]])[0]

        chunks: list[dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            chunks.append(
                {
                    "id": chunk_id,
                    "text": documents[idx] if idx < len(documents) else "",
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "distance": distances[idx] if idx < len(distances) else None,
                }
            )
        return chunks

    def get_document_list(self) -> list[DocumentInfo]:
        """Mengembalikan daftar dokumen yang tersimpan beserta jumlah chunk.

        Returns:
            List DocumentInfo dengan nama file dan jumlah chunk per dokumen.
        """
        if self._in_memory:
            return [
                DocumentInfo(filename=nama, chunk_count=jumlah)
                for nama, jumlah in self._dokumen.items()
            ]

        try:
            hasil = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise ChromaDBError(f"Gagal mengambil daftar dokumen: {exc}") from exc

        rekap: dict[str, int] = {}
        for metadata in hasil.get("metadatas", []) or []:
            filename = (metadata or {}).get("filename", "unknown")
            rekap[filename] = rekap.get(filename, 0) + 1

        return [
            DocumentInfo(filename=nama, chunk_count=jumlah)
            for nama, jumlah in sorted(rekap.items())
        ]

    def delete_all(self) -> bool:
        """Menghapus semua data dari ChromaDB.

        Returns:
            True jika penghapusan berhasil, False jika gagal.
        """
        if self._in_memory:
            self._dokumen.clear()
            self._chunks.clear()
            return True

        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return True
        except Exception as exc:
            raise ChromaDBError(f"Gagal menghapus data ChromaDB: {exc}") from exc

    def get_chunk_count(self) -> int:
        """Mengembalikan total jumlah chunk yang tersimpan.

        Returns:
            Total jumlah chunk di seluruh dokumen.
        """
        if self._in_memory:
            return sum(self._dokumen.values())

        try:
            return int(self._collection.count())
        except Exception as exc:
            raise ChromaDBError(f"Gagal menghitung jumlah chunk: {exc}") from exc

    def _build_chunk_id(self, metadata: dict[str, Any], text: str, index: int) -> str:
        basis = json.dumps(
            {
                "filename": metadata.get("filename", "unknown"),
                "position": metadata.get("position", ""),
                "chunk_index": metadata.get("chunk_index", index),
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()

    def _embed_text(self, text: str) -> list[float]:
        if self.embedding_function is not None:
            hasil = self.embedding_function(text)
            if isinstance(hasil, list):
                return hasil

        import urllib.error
        import urllib.request

        payload = json.dumps({"model": self.embedding_model, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
                embedding = data.get("embedding")
                if not embedding:
                    raise OllamaConnectionError("Ollama tidak mengembalikan embedding.")
                return embedding
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            if exc.code == 404 or "model" in body.lower():
                raise OllamaModelNotFoundError(
                    f"Model embedding '{self.embedding_model}' belum tersedia. Jalankan: ollama pull {self.embedding_model}"
                ) from exc
            raise OllamaConnectionError(
                f"Gagal mengakses Ollama embeddings API: {body or exc.reason}"
            ) from exc
        except Exception as exc:
            raise OllamaConnectionError(f"Gagal menghubungi Ollama: {exc}") from exc
