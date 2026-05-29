"""
TextLoader — load documents from any source and chunk them.

Supported sources
-----------------
  file path        : any .txt / .md / .json / .jsonl / .csv
  directory        : loads all .txt files recursively
  HuggingFace name : e.g. "wikitext", "wikipedia", "bookcorpus" (needs `datasets`)
  list of strings  : documents already in memory

Usage
-----
  from data.loader import TextLoader

  # From a local file
  docs = TextLoader.from_file("my_book.txt", chunk_size=128, overlap=32)

  # From a directory of .txt files
  docs = TextLoader.from_dir("./corpus/", chunk_size=128, overlap=32)

  # From HuggingFace datasets (requires: pip install datasets)
  docs = TextLoader.from_hf("wikitext", split="train",
                             subset="wikitext-2-raw-v1", chunk_size=128)

  # From raw strings
  docs = TextLoader.from_strings(["doc1 text...", "doc2 text..."], chunk_size=128)

Each loader returns List[List[str]] — a list of documents, each a list of chunks.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries. Keeps the delimiter."""
    pattern = r'(?<=[.!?])\s+'
    parts = re.split(pattern, text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_by_words(
    text: str,
    chunk_size: int = 128,
    overlap: int = 32,
    min_chunk_words: int = 16,
) -> List[str]:
    """
    Split text into overlapping word-level chunks.

    chunk_size : target number of words per chunk
    overlap    : word overlap between consecutive chunks
    """
    words = text.split()
    if len(words) < min_chunk_words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words) - chunk_size + 1, step):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
    # Include tail if it has enough content
    tail_start = ((len(words) - chunk_size) // step + 1) * step
    if tail_start < len(words) and len(words) - tail_start >= min_chunk_words:
        chunks.append(" ".join(words[tail_start:]))
    return chunks


def _read_file(path: str) -> str:
    """Read a text file, handling common formats."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8", errors="replace") as f:
        if ext == ".jsonl":
            texts = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Try common field names
                for key in ("text", "content", "body", "document"):
                    if key in obj and isinstance(obj[key], str):
                        texts.append(obj[key])
                        break
            return "\n\n".join(texts)
        elif ext == ".json":
            data = json.load(f)
            if isinstance(data, list):
                texts = []
                for item in data:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        for key in ("text", "content", "body"):
                            if key in item:
                                texts.append(str(item[key]))
                                break
                return "\n\n".join(texts)
            return str(data)
        else:
            return f.read()


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class TextLoader:

    @staticmethod
    def from_file(
        path: str,
        chunk_size: int = 128,
        overlap: int = 32,
        min_doc_words: int = 64,
    ) -> List[List[str]]:
        """Load one file as one document, chunked."""
        text = _read_file(path)
        # Split on blank lines to get natural paragraphs / sections as docs
        raw_docs = [d.strip() for d in re.split(r'\n\s*\n', text) if d.strip()]
        docs = []
        for raw in raw_docs:
            if len(raw.split()) < min_doc_words:
                continue
            chunks = chunk_by_words(raw, chunk_size, overlap)
            if len(chunks) >= 2:
                docs.append(chunks)
        if not docs:
            # Fall back: treat whole file as one document
            chunks = chunk_by_words(text, chunk_size, overlap)
            if len(chunks) >= 2:
                docs = [chunks]
        return docs

    @staticmethod
    def from_dir(
        directory: str,
        extensions: Tuple[str, ...] = (".txt", ".md"),
        chunk_size: int = 128,
        overlap: int = 32,
        max_files: Optional[int] = None,
    ) -> List[List[str]]:
        """Recursively load all matching files from a directory."""
        docs: List[List[str]] = []
        count = 0
        for root, _, files in os.walk(directory):
            for fname in sorted(files):
                if not fname.endswith(extensions):
                    continue
                path = os.path.join(root, fname)
                docs.extend(TextLoader.from_file(path, chunk_size, overlap))
                count += 1
                if max_files and count >= max_files:
                    return docs
        return docs

    @staticmethod
    def from_strings(
        texts: List[str],
        chunk_size: int = 128,
        overlap: int = 32,
        min_doc_words: int = 64,
    ) -> List[List[str]]:
        """Load from an in-memory list of document strings."""
        docs = []
        for text in texts:
            if len(text.split()) < min_doc_words:
                continue
            chunks = chunk_by_words(text, chunk_size, overlap)
            if len(chunks) >= 2:
                docs.append(chunks)
        return docs

    @staticmethod
    def from_hf(
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
        text_field: str = "text",
        chunk_size: int = 128,
        overlap: int = 32,
        max_docs: Optional[int] = None,
        min_doc_words: int = 64,
    ) -> List[List[str]]:
        """
        Load from a HuggingFace dataset.

        Requires: pip install datasets

        Common dataset names and subsets
        ---------------------------------
        wikitext       subset="wikitext-2-raw-v1"  (small, clean Wikipedia)
        wikitext       subset="wikitext-103-raw-v1" (large)
        wikipedia      subset="20220301.en"          (full Wikipedia, large)
        bookcorpus     (no subset needed)
        openwebtext    (no subset needed)
        """
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError(
                "HuggingFace `datasets` not installed. "
                "Run: pip install datasets"
            )

        load_kwargs: Dict = {"split": split}
        if subset:
            load_kwargs["name"] = subset
        if max_docs:
            load_kwargs["streaming"] = True

        ds = load_dataset(dataset_name, **load_kwargs)

        docs: List[List[str]] = []
        count = 0
        for item in ds:
            text = item.get(text_field, "") if isinstance(item, dict) else str(item)
            if not text or len(text.split()) < min_doc_words:
                continue
            chunks = chunk_by_words(text, chunk_size, overlap)
            if len(chunks) >= 2:
                docs.append(chunks)
                count += 1
            if max_docs and count >= max_docs:
                break

        return docs

    @staticmethod
    def stats(docs: List[List[str]]) -> Dict[str, float]:
        """Print stats about a loaded dataset."""
        n_docs   = len(docs)
        n_chunks = sum(len(d) for d in docs)
        lengths  = [len(c.split()) for d in docs for c in d]
        return {
            "n_docs":         n_docs,
            "n_chunks":       n_chunks,
            "chunks_per_doc": n_chunks / max(n_docs, 1),
            "words_per_chunk": sum(lengths) / max(len(lengths), 1),
            "total_words":    sum(lengths),
        }
