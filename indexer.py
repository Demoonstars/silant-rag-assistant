import warnings
warnings.filterwarnings("ignore")

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from parser import parse_all_manuals, parse_single_pdf, update_models_registry

load_dotenv()

CHUNKS_PATH = "data/parsed_chunks.json"
CHROMA_DIR = "data/chroma_db"

_EMBEDDINGS_INSTANCE = None


def sanitize_metadata(meta: dict) -> dict:
    """гарантируем, что в metadata для ChromaDB нет списков или словарей"""
    clean = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (list, tuple, set)):
            clean[k] = "; ".join(str(x) for x in v)
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean


def get_embeddings():
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        print(" -> Загружаем локальную модель векторизации (без лимитов API)...", flush=True)
        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _EMBEDDINGS_INSTANCE


def get_chroma_instance():
    embeddings = get_embeddings()
    client_settings = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=CHROMA_DIR
    )
    return Chroma(
        collection_name="silant_manuals",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        client_settings=client_settings
    )


def load_documents_from_json(json_path: str = CHUNKS_PATH) -> list[Document]:
    path = Path(json_path)
    if not path.exists():
        parse_all_manuals(output_json=json_path)

    with open(path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    # оставляем только те чанки, чьи PDF реально лежат в папке data/manuals
    existing_pdfs = {p.name for p in Path("data/manuals").glob("*.pdf")}
    docs = []
    for item in raw_chunks:
        src = item.get("metadata", {}).get("source_file", "")
        if existing_pdfs and src not in existing_pdfs:
            continue
        doc = Document(
            page_content=item["text"],
            metadata=sanitize_metadata(item.get("metadata", {}))
        )
        docs.append(doc)
    return docs


def remove_pdf_from_db(filename: str) -> int:
    """удаляет один пдф-файл с диска и мгновенно вычищает его чанки и модели из базы"""
    pdf_path = Path("data/manuals") / filename
    if pdf_path.exists():
        pdf_path.unlink()

    existing_pdfs = {p.name for p in Path("data/manuals").glob("*.pdf")}
    chunks_file = Path(CHUNKS_PATH)
    remaining_chunks = []
    removed_count = 0

    if chunks_file.exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        for ch in all_chunks:
            src = ch.get("metadata", {}).get("source_file", "")
            if src == filename or (existing_pdfs and src not in existing_pdfs):
                removed_count += 1
            else:
                remaining_chunks.append(ch)

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(remaining_chunks, f, ensure_ascii=False, indent=2)

    update_models_registry(remaining_chunks)

    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists() and any(chroma_path.iterdir()):
        try:
            vs = get_chroma_instance()
            vs._collection.delete(where={"source_file": filename})
        except Exception as e:
            print(f"Предупреждение при очистке ChromaDB: {e}")

    print(f"[Удаление] Файл {filename} удален, убрано страниц из базы: {removed_count}")
    return removed_count


def index_specific_pdfs(target_filenames: list[str], progress_callback=None) -> tuple[int, list[str]]:
    """принудительно читает, выявляет модели и индексирует загруженные пользователем PDF-файлы"""
    manuals_dir = Path("data/manuals")
    existing_pdfs = {p.name for p in manuals_dir.glob("*.pdf")}

    chunks_file = Path(CHUNKS_PATH)
    kept_chunks = []
    if chunks_file.exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            old_chunks = json.load(f)
        # убираем старые чанки перезагружаемых файлов (и любых удаленных ранее файлов)
        for ch in old_chunks:
            src = ch.get("metadata", {}).get("source_file", "")
            if src not in target_filenames and src in existing_pdfs:
                kept_chunks.append(ch)

    vs = get_chroma_instance()
    for fname in target_filenames:
        try:
            vs._collection.delete(where={"source_file": fname})
        except Exception:
            pass

    new_chunks = []
    discovered_models = []
    for idx_f, fname in enumerate(target_filenames, 1):
        pdf_path = manuals_dir / fname
        if not pdf_path.exists():
            continue
        if progress_callback:
            progress_callback(0.1, f"Чтение и выявление моделей: {fname} ({idx_f}/{len(target_filenames)})...")
        ch_list, file_models = parse_single_pdf(pdf_path)
        if not ch_list:
            pdf_path.unlink(missing_ok=True)
            raise ValueError(f"В файле «{fname}» отсутствует текстовый слой (возможно, это скан-копия без OCR).")
        new_chunks.extend(ch_list)
        for m in file_models:
            if m not in discovered_models:
                discovered_models.append(m)

    new_docs = [
        Document(page_content=c["text"], metadata=sanitize_metadata(c.get("metadata", {})))
        for c in new_chunks
    ]

    batch_size = 50
    total = len(new_docs)
    for i in range(0, total, batch_size):
        batch = new_docs[i : i + batch_size]
        done = min(i + batch_size, total)
        frac = done / total
        pct = int(frac * 100)
        vs.add_documents(batch)
        print(f" -> Добавлено в векторную базу: {done} из {total} стр. ({pct}%)")
        if progress_callback:
            progress_callback(frac, f"Индексация страниц и моделей: {pct}%")

    all_chunks = kept_chunks + new_chunks
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    update_models_registry(all_chunks)
    return len(new_docs), discovered_models


def add_new_pdfs_incremental(progress_callback=None) -> int:
    """проверяет все PDF в папке data/manuals и доиндексирует те, которых нет в базе или реестре моделей"""
    chunks_file = Path(CHUNKS_PATH)
    existing_chunks = []
    if chunks_file.exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)

    indexed_files = {ch.get("metadata", {}).get("source_file") for ch in existing_chunks}
    pdf_files = sorted(Path("data/manuals").glob("*.pdf"))
    new_names = [p.name for p in pdf_files if p.name not in indexed_files]

    if not new_names:
        update_models_registry(existing_chunks)
        return 0

    added_count, _ = index_specific_pdfs(new_names, progress_callback=progress_callback)
    return added_count


def build_vector_db(rebuild_pdfs: bool = False, progress_callback=None):
    """полная пересборка базы за несколько секунд на локальных эмбеддингах"""
    if rebuild_pdfs:
        if progress_callback:
            progress_callback(0.05, "Обновление базы знаний: 5%")
        parse_all_manuals(output_json=CHUNKS_PATH)

    docs = load_documents_from_json(CHUNKS_PATH)
    if not docs:
        print("[ОШИБКА] Нет данных для индексации")
        return None

    vectorstore = get_chroma_instance()
    try:
        existing_ids = vectorstore._collection.get()["ids"]
        if existing_ids:
            vectorstore._collection.delete(ids=existing_ids)
    except Exception:
        pass

    batch_size = 50
    total = len(docs)

    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        done = min(i + batch_size, total)
        frac = done / total
        pct = int(frac * 100)

        vectorstore.add_documents(batch)
        print(f" -> Записано в векторную базу: {done} из {total} стр. ({pct}%)")
        if progress_callback:
            progress_callback(frac, f"Обновление базы знаний: {pct}%")

    print(f"[ОК] Векторная база успешно собрана ({total} стр.)")
    return vectorstore


def get_hybrid_retriever(top_k: int = 14):
    docs = load_documents_from_json(CHUNKS_PATH)
    chroma_path = Path(CHROMA_DIR)

    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        vectorstore = build_vector_db(rebuild_pdfs=False)
    else:
        vectorstore = get_chroma_instance()

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid_retriever


if __name__ == "__main__":
    build_vector_db(rebuild_pdfs=True)