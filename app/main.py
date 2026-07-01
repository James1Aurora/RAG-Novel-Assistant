"""
RAG Novel Assistant — Entry Point Streamlit.

Aplikasi ini memungkinkan penulis novel mengupload dokumen (PDF, TXT, DOCX)
dan mengajukan pertanyaan tentang isi novel menggunakan RAG berbasis Ollama lokal.
"""

import os
import tempfile
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.exceptions import (
    ChromaDBError,
    DuplicateDocumentError,
    EmptyDocumentError,
    EmptyVectorStoreError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    ValidationError,
)
from app.rag_pipeline import ProcessResult, QueryResult, RAGPipeline, SourceReference
from app.vector_store import VectorStoreManager

# ─────────────────────────────────────────────
# Konfigurasi awal
# ─────────────────────────────────────────────
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3")
MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", "50"))

MODEL_TERSEDIA = ["llama3", "mistral"]

# ─────────────────────────────────────────────
# Inisialisasi komponen (singleton per sesi)
# ─────────────────────────────────────────────

@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Menginisialisasi VectorStoreManager sebagai singleton per sesi Streamlit."""
    return VectorStoreManager(persist_directory=CHROMA_PERSIST_DIR)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """Menginisialisasi RAGPipeline sebagai singleton per sesi Streamlit."""
    return RAGPipeline(
        vector_store=get_vector_store(),
        ollama_base_url=OLLAMA_BASE_URL,
    )


# ─────────────────────────────────────────────
# Inisialisasi session state
# ─────────────────────────────────────────────

def init_session_state() -> None:
    """Menginisialisasi semua variabel session state dengan nilai default."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (pertanyaan, QueryResult)
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    if "documents_ready" not in st.session_state:
        st.session_state.documents_ready = get_vector_store().get_chunk_count() > 0
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "show_reset_dialog" not in st.session_state:
        st.session_state.show_reset_dialog = False
    if "pesan_upload" not in st.session_state:
        st.session_state.pesan_upload = None  # (tipe, teks) — tipe: "success"/"error"/"info"


# ─────────────────────────────────────────────
# Fungsi tambah ke riwayat chat (FIFO, maks 50)
# ─────────────────────────────────────────────

def tambah_ke_riwayat(pertanyaan: str, hasil: QueryResult) -> None:
    """Menambahkan pasangan pertanyaan-jawaban ke riwayat chat dengan batas FIFO.

    Args:
        pertanyaan: Pertanyaan yang diajukan pengguna.
        hasil: QueryResult dari RAG pipeline.
    """
    st.session_state.chat_history.append((pertanyaan, hasil))
    # Hapus pasangan terlama jika melebihi batas
    if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]


# ─────────────────────────────────────────────
# Handle Upload
# ─────────────────────────────────────────────

def handle_upload(uploaded_file, pipeline: RAGPipeline) -> None:
    """Menangani upload file dan memanggil pipeline untuk pemrosesan.

    Menyimpan file sementara, memproses melalui pipeline, dan memperbarui
    pesan status upload di session state.

    Args:
        uploaded_file: File yang diupload melalui st.file_uploader.
        pipeline: Instance RAGPipeline untuk memproses dokumen.
    """
    if uploaded_file is None:
        return

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        hasil = pipeline.load_and_process_document(
            file_path=tmp_path,
            filename=uploaded_file.name,
        )
        st.session_state.pesan_upload = (
            "success",
            f"✅ {hasil.message}",
        )
        st.session_state.documents_ready = True

        if hasil.chunks_failed > 0:
            st.session_state.pesan_upload = (
                "warning",
                f"⚠️ {hasil.message} ({hasil.chunks_failed} chunk gagal diproses.)",
            )

    except DuplicateDocumentError as e:
        st.session_state.pesan_upload = ("info", f"ℹ️ {e}")
    except ValidationError as e:
        st.session_state.pesan_upload = ("error", f"❌ {e}")
    except EmptyDocumentError as e:
        st.session_state.pesan_upload = ("error", f"❌ {e}")
    except OllamaConnectionError as e:
        st.session_state.pesan_upload = (
            "error",
            f"❌ Ollama tidak dapat dijangkau. Pastikan Ollama sedang berjalan di {OLLAMA_BASE_URL}.",
        )
    except Exception as e:
        st.session_state.pesan_upload = ("error", f"❌ Terjadi kesalahan: {e}")
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# Handle Query
# ─────────────────────────────────────────────

def handle_query(question: str, model: str, pipeline: RAGPipeline) -> None:
    """Menangani query pengguna dan menambahkan hasilnya ke riwayat chat.

    Args:
        question: Pertanyaan dari pengguna.
        model: Nama model Ollama yang aktif.
        pipeline: Instance RAGPipeline.
    """
    if not question.strip():
        return

    try:
        hasil = pipeline.query(question=question, model=model)
        tambah_ke_riwayat(question, hasil)
    except EmptyVectorStoreError:
        hasil_kosong = QueryResult(
            answer="",
            sources=[],
            error="Belum ada dokumen yang diupload. Silakan upload novel terlebih dahulu.",
        )
        tambah_ke_riwayat(question, hasil_kosong)
    except OllamaConnectionError:
        hasil_error = QueryResult(
            answer="",
            sources=[],
            error=f"LLM tidak dapat dijangkau. Pastikan Ollama berjalan di {OLLAMA_BASE_URL}.",
        )
        tambah_ke_riwayat(question, hasil_error)
    except OllamaModelNotFoundError as e:
        hasil_error = QueryResult(
            answer="",
            sources=[],
            error=str(e),
        )
        tambah_ke_riwayat(question, hasil_error)
    except Exception as e:
        hasil_error = QueryResult(
            answer="",
            sources=[],
            error=f"Terjadi kesalahan: {e}",
        )
        tambah_ke_riwayat(question, hasil_error)


# ─────────────────────────────────────────────
# Render: Daftar Dokumen
# ─────────────────────────────────────────────

def render_document_list(vector_store: VectorStoreManager) -> None:
    """Menampilkan daftar dokumen yang telah diproses di sidebar.

    Args:
        vector_store: Instance VectorStoreManager.
    """
    dokumen = vector_store.get_document_list()

    if not dokumen:
        st.caption("Belum ada dokumen yang diupload.")
        return

    st.markdown("**Dokumen tersimpan:**")
    for doc in dokumen:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"📄 `{doc.filename}`")
        with col2:
            st.caption(f"{doc.chunk_count} chunk")


# ─────────────────────────────────────────────
# Render: Sidebar
# ─────────────────────────────────────────────

def render_sidebar(pipeline: RAGPipeline, vector_store: VectorStoreManager) -> None:
    """Merender sidebar dengan komponen upload, pilihan model, dan manajemen dokumen.

    Args:
        pipeline: Instance RAGPipeline untuk pemrosesan dokumen.
        vector_store: Instance VectorStoreManager untuk daftar dokumen.
    """
    with st.sidebar:
        st.title("📚 RAG Novel Assistant")
        st.markdown("---")

        # ── Upload Dokumen ──────────────────────
        st.subheader("📤 Upload Novel")
        st.caption("Format yang didukung: PDF, TXT, DOCX (maks. 50 MB)")

        uploaded_file = st.file_uploader(
            label="Pilih file novel",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
        )

        if st.button("Proses Dokumen", use_container_width=True, type="primary"):
            if uploaded_file is None:
                st.warning("Pilih file terlebih dahulu.")
            else:
                with st.spinner("Memproses dokumen..."):
                    handle_upload(uploaded_file, pipeline)
                st.rerun()

        # Tampilkan pesan hasil upload
        if st.session_state.pesan_upload:
            tipe, teks = st.session_state.pesan_upload
            if tipe == "success":
                st.success(teks)
            elif tipe == "error":
                st.error(teks)
            elif tipe == "info":
                st.info(teks)
            elif tipe == "warning":
                st.warning(teks)

        st.markdown("---")

        # ── Daftar Dokumen ──────────────────────
        st.subheader("📋 Dokumen Tersimpan")
        render_document_list(vector_store)

        st.markdown("---")

        # ── Pilihan Model ───────────────────────
        st.subheader("🤖 Model Ollama")
        model_dipilih = st.selectbox(
            label="Pilih model",
            options=MODEL_TERSEDIA,
            index=MODEL_TERSEDIA.index(st.session_state.selected_model)
            if st.session_state.selected_model in MODEL_TERSEDIA
            else 0,
            label_visibility="collapsed",
        )
        if model_dipilih != st.session_state.selected_model:
            st.session_state.selected_model = model_dipilih
            st.rerun()

        st.caption(f"Model aktif: `{st.session_state.selected_model}`")

        st.markdown("---")

        # ── Reset ───────────────────────────────
        st.subheader("🗑️ Reset Data")
        st.caption("Hapus semua dokumen dan riwayat percakapan.")

        if st.button("Reset Semua Data", use_container_width=True, type="secondary"):
            st.session_state.show_reset_dialog = True
            st.rerun()


# ─────────────────────────────────────────────
# Render: Dialog Reset
# ─────────────────────────────────────────────

def render_dialog_reset(vector_store: VectorStoreManager) -> None:
    """Menampilkan panel konfirmasi reset di area utama.

    Dipanggil ketika show_reset_dialog == True. Menggantikan area chat
    dengan panel konfirmasi agar pengguna tidak dapat berinteraksi lain
    sebelum membuat pilihan.

    Args:
        vector_store: Instance VectorStoreManager untuk operasi penghapusan.
    """
    st.warning(
        "⚠️ Tindakan ini akan **menghapus semua dokumen** dari vector store "
        "dan **mengosongkan riwayat percakapan**. Tindakan ini tidak dapat dibatalkan."
    )
    st.markdown(
        f"Total chunk yang akan dihapus: **{vector_store.get_chunk_count()}**"
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("Batalkan", use_container_width=True):
            st.session_state.show_reset_dialog = False
            st.rerun()
    with col3:
        if st.button("Ya, Hapus Semua", use_container_width=True, type="primary"):
            try:
                berhasil = vector_store.delete_all()
                if berhasil:
                    st.session_state.chat_history = []
                    st.session_state.documents_ready = False
                    st.session_state.pesan_upload = None
                    st.session_state.show_reset_dialog = False
                    st.rerun()
                else:
                    st.error("Penghapusan gagal. Vector store tetap tidak berubah.")
            except ChromaDBError:
                st.error("Penghapusan gagal. Tidak dapat mengakses vector store.")


# ─────────────────────────────────────────────
# Render: Area Chat
# ─────────────────────────────────────────────

def render_chat_area() -> None:
    """Merender area chat utama dengan riwayat percakapan.

    Menampilkan semua pasangan pertanyaan-jawaban dari session state,
    termasuk sumber referensi di bawah setiap jawaban.
    """
    if not st.session_state.chat_history:
        st.markdown(
            "<div style='text-align:center; color:#888; padding: 60px 20px;'>"
            "💬 Ajukan pertanyaan tentang novel yang telah Anda upload.<br>"
            "<small>Contoh: <i>Siapa karakter utama di bab ini?</i></small>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for pertanyaan, hasil in st.session_state.chat_history:
        # Balon chat pengguna
        with st.chat_message("user"):
            st.markdown(pertanyaan)

        # Balon chat asisten
        with st.chat_message("assistant"):
            if hasil.error:
                st.error(hasil.error)
            else:
                st.markdown(hasil.answer)

                # Tampilkan sumber referensi
                if hasil.sources:
                    with st.expander("📎 Sumber Referensi", expanded=False):
                        for i, sumber in enumerate(hasil.sources, 1):
                            if sumber.position:
                                label = f"{sumber.filename} — {sumber.position}"
                            else:
                                label = sumber.filename
                            st.markdown(f"**{i}.** `{label}`")
                else:
                    st.caption("_Tidak ada sumber referensi yang tersedia._")


# ─────────────────────────────────────────────
# Render: Input Pertanyaan
# ─────────────────────────────────────────────

def render_input_pertanyaan(pipeline: RAGPipeline) -> None:
    """Merender input pertanyaan di bagian bawah area chat.

    Input dinonaktifkan jika belum ada dokumen yang diupload.
    Menampilkan spinner loading selama pipeline memproses query.

    Args:
        pipeline: Instance RAGPipeline untuk menjalankan query.
    """
    if not st.session_state.documents_ready:
        st.info(
            "📤 Upload novel terlebih dahulu melalui sidebar untuk mulai bertanya."
        )
        # Input nonaktif sebagai placeholder visual
        st.chat_input(
            placeholder="Upload dokumen novel terlebih dahulu...",
            disabled=True,
        )
        return

    pertanyaan = st.chat_input(
        placeholder="Tanyakan sesuatu tentang novel Anda...",
    )

    if pertanyaan:
        with st.spinner("Mencari jawaban..."):
            handle_query(
                question=pertanyaan,
                model=st.session_state.selected_model,
                pipeline=pipeline,
            )
        st.rerun()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    """Entry point utama aplikasi Streamlit RAG Novel Assistant."""
    st.set_page_config(
        page_title="RAG Novel Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    vector_store = get_vector_store()
    pipeline = get_pipeline()

    # Sinkronisasi documents_ready dengan state vector store aktual
    st.session_state.documents_ready = vector_store.get_chunk_count() > 0

    # Render sidebar
    render_sidebar(pipeline, vector_store)

    # Panel konfirmasi reset (menggantikan area utama jika flag aktif)
    if st.session_state.show_reset_dialog:
        st.markdown(
            "<h2 style='margin-bottom: 0;'>🗑️ Konfirmasi Reset</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        render_dialog_reset(vector_store)
        st.stop()  # hentikan render di sini, jangan tampilkan area chat

    # ── Header area utama ──────────────────────
    st.markdown(
        "<h2 style='margin-bottom: 0;'>💬 Tanya Jawab Novel</h2>",
        unsafe_allow_html=True,
    )

    jumlah_doc = len(vector_store.get_document_list())
    total_chunk = vector_store.get_chunk_count()
    model_aktif = st.session_state.selected_model

    if jumlah_doc > 0:
        st.caption(
            f"📚 {jumlah_doc} dokumen tersimpan · "
            f"🔢 {total_chunk} chunk · "
            f"🤖 Model: `{model_aktif}`"
        )
    else:
        st.caption(f"🤖 Model: `{model_aktif}` · Belum ada dokumen yang diupload.")

    st.markdown("---")

    # ── Area chat ─────────────────────────────
    render_chat_area()

    # ── Input pertanyaan ───────────────────────
    render_input_pertanyaan(pipeline)


if __name__ == "__main__":
    main()
