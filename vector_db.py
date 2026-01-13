"""
vector_db.py — 2025-compatible, no external MultiQuery dependency

- Chroma vector store with incremental sync (hash-based)
- OpenAI embeddings (text-embedding-3-small)
- Manual GPT-assisted query expansion (RRF merge) → better recall without MultiQueryRetriever
- Modes:
    * "hybrid"  : semantic + GPT expansions (recommended)
    * "tabular" : same but filtered to CSV/XLSX sources
    * "semantic": plain semantic MMR (no expansions)
- Public API returns a retriever object exposing .invoke(query) -> List[Document]
- Test block uses .invoke()
"""

import os
import json
import hashlib
import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# ---------------------- LangChain integrations ----------------------
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
)

# ============================== Paths =================================
def get_base_path():
    """Return (base_data_folder, persist_directory, index_csv_path)."""
    if os.path.exists("/content/drive/MyDrive"):
        base_data = "/content/drive/MyDrive/Synthetic_Data"
        persist_dir = "/content/drive/MyDrive/Chartbot/Kestrel"
        index_path = "/content/drive/MyDrive/Chartbot/metadata_index.csv"
    else:
        base_data = "./Real_Data"
        persist_dir = "./Kestrel"
        index_path = "./metadata_index.csv"
    return base_data, persist_dir, index_path

# ======================= Load & Index Documents =======================
def load_and_index(folder_path: str, index_path: str):
    """Recursively load supported files and write/refresh a metadata CSV."""
    SUPPORTED = {
        "pdf":  UnstructuredPDFLoader,
        "doc":  UnstructuredWordDocumentLoader,
        "docx": UnstructuredWordDocumentLoader,
        "ppt":  UnstructuredPowerPointLoader,
        "pptx": UnstructuredPowerPointLoader,
        "csv":  CSVLoader,
        "xlsx": UnstructuredExcelLoader,
        "md":   UnstructuredMarkdownLoader,
    }

    all_docs, index_records = [], []
    print(f"\nScanning directory: {folder_path}")

    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = file.lower().split(".")[-1]
            if ext not in SUPPORTED:
                continue
            full_path = os.path.join(root, file)
            try:
                loader = SUPPORTED[ext](full_path)
                docs = loader.load()
                rel_path = os.path.relpath(full_path, folder_path)
                for d in docs:
                    d.metadata["source"] = rel_path
                    d.metadata["file_type"] = ext
                    index_records.append({
                        "filename": file,
                        "relative_path": rel_path,
                        "ext": ext,
                        "length": len(d.page_content),
                        "snippet": d.page_content[:200].replace("\n", " "),
                    })
                all_docs.extend(docs)
            except Exception as e:
                print(f"Skipping {file}: {e}")

    index_df = pd.DataFrame(index_records)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    if index_df.empty:
        print("⚠️ No supported documents found to index.")
        index_df = pd.DataFrame(
            columns=["filename", "relative_path", "ext", "length", "snippet"]
        )

    if not os.path.exists(index_path) or os.path.getsize(index_path) == 0:
        index_df.to_csv(index_path, index=False)
        print(f"Metadata index created at {index_path}")
    else:
        try:
            old = pd.read_csv(index_path)
            if not old.equals(index_df):
                index_df.to_csv(index_path, index=False)
                print(f"Metadata index updated at {index_path}")
            else:
                print("No metadata changes detected — reusing existing index.")
        except pd.errors.EmptyDataError:
            index_df.to_csv(index_path, index=False)
            print(f"Existing metadata file was empty — recreated {index_path}")

    print(f"Indexed {len(index_records)} document chunks.")
    return all_docs, index_df

# ===================== Vector DB (Incremental Sync) ====================
def compute_file_hash(path):
    """Return short MD5 hash for a file; 'missing' if unreadable."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    except Exception:
        return "missing"

def get_or_update_vectorstore_sync(docs, index_df, base_data_folder, persist_directory):
    """
    Create or update a Chroma vector DB incrementally using file hashes.
    """
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    meta_path = os.path.join(persist_directory, "embedding_meta.json")
    os.makedirs(persist_directory, exist_ok=True)

    chroma_path = os.path.join(persist_directory, "chroma.sqlite3")
    if os.path.exists(chroma_path):
        print(f"Existing Chroma DB found at {persist_directory}. Checking for updates...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        existing_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    else:
        print("No existing embeddings found. Creating new Chroma DB...")
        vectorstore = None
        existing_meta = {}

    current_meta = {}
    for _, row in index_df.iterrows():
        rel_path = row.get("relative_path")
        if not isinstance(rel_path, str):
            continue
        full_path = os.path.join(base_data_folder, rel_path)
        current_meta[rel_path] = {
            "hash": compute_file_hash(full_path),
            "length": int(row.get("length", 0)),
        }

    current_files = set(current_meta.keys())
    old_files = set(existing_meta.keys())
    new_or_changed = [
        f for f in current_files
        if f not in old_files or current_meta[f]["hash"] != existing_meta.get(f, {}).get("hash")
    ]
    deleted = list(old_files - current_files)

    if deleted and vectorstore:
        print(f"Removing {len(deleted)} deleted file(s)...")
        try:
            # Filter delete on metadata "source"
            vectorstore._collection.delete(where={"source": {"$in": deleted}})
        except Exception as e:
            print("Could not remove some embeddings:", e)

    if new_or_changed:
        changed_docs = [d for d in docs if d.metadata.get("source") in new_or_changed]
        print(f"Re-embedding {len(changed_docs)} new/changed file(s)...")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=changed_docs,
                embedding=embedding_model,
                persist_directory=persist_directory
            )
        else:
            vectorstore.add_documents(changed_docs)
        print("Vectorstore updated successfully.")
    else:
        if vectorstore is None:
            if docs:
                vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=embedding_model,
                    persist_directory=persist_directory
                )
                print("Vectorstore created from full corpus.")
            else:
                print("⚠️ No documents to embed.")
        else:
            print("No new or changed files detected — using existing embeddings.")

    with open(meta_path, "w") as f:
        json.dump(current_meta, f, indent=2)
    print("Metadata synced successfully.")
    return vectorstore

# ======================== Expansion + Fusion ===========================
def _generate_expansions(user_query: str, n: int = 3) -> List[str]:
    """
    Generate n reformulated queries with ChatOpenAI (aggressive recall).
    If the LLM fails, return the original query only.
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        system = (
            "You expand a user's search query into semantically different variants that might retrieve "
            "complementary documents. Be concise and avoid punctuation-heavy rewrites."
        )
        user = (
            f"Original query:\n{user_query}\n\n"
            f"Return exactly {n} different reformulations, each on a new line, no numbering."
        )
        msg = llm.invoke([{"role": "system", "content": system},
                          {"role": "user", "content": user}])
        lines = [l.strip() for l in (msg.content or "").split("\n") if l.strip()]
        # keep only n lines
        return lines[:n] if lines else [user_query]
    except Exception:
        return [user_query]

def _doc_key(doc: Document) -> Tuple[str, Any]:
    """A stable key for a Document to deduplicate/score across queries."""
    src = str(doc.metadata.get("source", ""))
    page = doc.metadata.get("page", doc.metadata.get("page_number", None))
    return (src, page)

def _rrf_fuse(rank_lists: List[List[Document]], k: int = 5, k_rrf: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.
    rank_lists: list of result lists (one per query/expansion)
    k: final top-k to return
    """
    scores = defaultdict(float)
    first_seen: Dict[Tuple[str, Any], Document] = {}
    for results in rank_lists:
        for rank, doc in enumerate(results, start=1):
            key = _doc_key(doc)
            if key not in first_seen:
                first_seen[key] = doc
            scores[key] += 1.0 / (k_rrf + rank)
    # sort by fused score desc
    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [first_seen[k_] for k_ in ranked_keys[:k]]

# ======================= Expanded Retriever ===========================
class ExpandedRetriever:
    """
    A retriever that:
      - Applies optional metadata filter (e.g., only CSV/XLSX)
      - Uses MMR search
      - Expands the query with GPT (n=3) for aggressive recall
      - Fuses results via Reciprocal Rank Fusion
      - Exposes .invoke(query) to match latest retriever API expectations
    """
    def __init__(self, vectorstore: Chroma, k: int = 5, metadata_filter: Dict[str, Any] | None = None,
                 do_expand: bool = True):
        self.vs = vectorstore
        self.k = k
        self.filter = metadata_filter
        self.do_expand = do_expand

    def _search_once(self, q: str, fetch_k: int = 16, lambda_mult: float = 0.7) -> List[Document]:
        # Prefer MMR to improve diversity
        try:
            return self.vs.max_marginal_relevance_search(
                q, k=self.k, fetch_k=fetch_k, lambda_mult=lambda_mult,
                filter=self.filter
            )
        except TypeError:
            # Some older bindings take different kwargs
            return self.vs.max_marginal_relevance_search(q, k=self.k)

    def invoke(self, query: str) -> List[Document]:
        if not self.do_expand:
            return self._search_once(query)

        # Generate n expansions and search all
        expansions = _generate_expansions(query, n=3)
        all_ranked = []
        # Include original query at the front
        for q in [query] + expansions:
            try:
                res = self._search_once(q)
            except Exception:
                res = []
            all_ranked.append(res)

        # Fuse & return top-k
        fused = _rrf_fuse(all_ranked, k=self.k)
        return fused

# ========================= Builders / Public API ======================
def build_hybrid_retriever(vectorstore, k=5) -> ExpandedRetriever:
    """Hybrid = semantic MMR + GPT expansions (recommended)."""
    return ExpandedRetriever(vectorstore, k=k, metadata_filter=None, do_expand=True)

def build_tabular_retriever(vectorstore, k=5) -> ExpandedRetriever:
    """Tabular = same as hybrid, but restricted to CSV/XLSX."""
    tab_filter = {"file_type": {"$in": ["csv", "xlsx"]}}
    return ExpandedRetriever(vectorstore, k=k, metadata_filter=tab_filter, do_expand=True)

def build_semantic_retriever(vectorstore, k=5) -> ExpandedRetriever:
    """Semantic-only = MMR, no expansions."""
    return ExpandedRetriever(vectorstore, k=k, metadata_filter=None, do_expand=False)

def get_retriever(retriever_type="hybrid"):
    """
    Return a retriever with .invoke(query) -> List[Document]
    retriever_type: "hybrid" | "tabular" | "semantic"
    """
    base_data, persist_dir, index_path = get_base_path()
    docs, index_df = load_and_index(base_data, index_path)
    vectorstore = get_or_update_vectorstore_sync(docs, index_df, base_data, persist_dir)
    if vectorstore is None:
        raise RuntimeError("No vectorstore created — ensure your data folder has supported files.")

    if retriever_type == "tabular":
        print("📊 Using TABULAR (CSV/XLSX) expanded retriever (k=5).")
        return build_tabular_retriever(vectorstore, k=5)
    elif retriever_type == "hybrid":
        print("🔎 Using HYBRID expanded retriever (k=5).")
        return build_hybrid_retriever(vectorstore, k=5)
    else:
        print("🧠 Using SEMANTIC-ONLY retriever (k=5).")
        return build_semantic_retriever(vectorstore, k=5)

# ============================== Tests =================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING RETRIEVER MODES (no MultiQuery dependency)")
    print("="*70)

    base_data, persist_dir, index_path = get_base_path()
    docs, index_df = load_and_index(base_data, index_path)
    vs = get_or_update_vectorstore_sync(docs, index_df, base_data, persist_dir)

    # --- Hybrid test ---
    print("\n📚 TEST 1: Hybrid (RAG, k=5)")
    print("-" * 70)
    r = build_hybrid_retriever(vs, k=5)
    q = "What should a consultant do in the first 48 hours?"
    results = r.invoke(q)
    for i, d in enumerate(results[:5], 1):
        print(f"\nResult {i}:")
        print(f"  Source: {d.metadata.get('source')}")
        print(f"  Type: {d.metadata.get('file_type', 'unknown')}")
        print(f"  Preview: {d.page_content[:150]}...")

    # --- Tabular test ---
    print("\n\n📊 TEST 2: Tabular (Visualization, k=5)")
    print("-" * 70)
    rt = build_tabular_retriever(vs, k=5)
    q2 = "Show me data about allocation percentages by role"
    results2 = rt.invoke(q2)
    for i, d in enumerate(results2[:5], 1):
        print(f"\nResult {i}:")
        print(f"  Source: {d.metadata.get('source')}")
        print(f"  Type: {d.metadata.get('file_type', 'unknown')}")
        print(f"  Preview: {d.page_content[:150]}...")

    print("\n" + "="*70)
    print("✅ Testing complete!")
    print("="*70)
