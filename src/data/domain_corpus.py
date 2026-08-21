"""Raw domain-text ingestion and chunking for Continued Pre-Training (CLM).

Loads every .txt file under a corpus directory, concatenates the tokenized
stream, and splits it into fixed-size, non-overlapping blocks so that the
causal-LM loss is computed densely across the domain corpus (standard CPT
data preparation, distinct from instruction-formatted SFT data).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CorpusStats:
    num_documents: int
    num_tokens: int
    num_blocks: int


def load_raw_documents(corpus_dir: Path) -> list[str]:
    """Read all .txt files in a directory (non-recursive is intentional: keeps
    the domain corpus layout explicit and auditable for a portfolio project)."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Domain corpus directory not found: {corpus_dir}")

    documents: list[str] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(text)

    if not documents:
        raise ValueError(f"No non-empty .txt documents found in {corpus_dir}")
    return documents


def chunk_token_ids(token_ids: list[int], block_size: int) -> list[list[int]]:
    """Split a flat token-id stream into contiguous blocks of `block_size`.

    Trailing tokens that don't fill a full block are dropped, matching the
    standard CLM pre-training recipe (see e.g. GPT-style pretraining pipelines).
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n_full_blocks = len(token_ids) // block_size
    return [
        token_ids[i * block_size : (i + 1) * block_size] for i in range(n_full_blocks)
    ]


def build_cpt_dataset(
    corpus_dir: Path,
    tokenizer: Any,
    block_size: int,
    val_split: float = 0.05,
) -> tuple[Any, Any, CorpusStats]:
    """Tokenize the domain corpus and return (train_dataset, val_dataset, stats)
    as Hugging Face `datasets.Dataset` objects with an `input_ids` column."""
    from datasets import Dataset

    documents = load_raw_documents(corpus_dir)
    joined = "\n\n".join(documents)
    token_ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
    blocks = chunk_token_ids(token_ids, block_size)

    if not blocks:
        raise ValueError(
            f"Corpus at {corpus_dir} produced fewer than {block_size} tokens; "
            "cannot form a single training block."
        )

    n_val = max(1, int(len(blocks) * val_split)) if len(blocks) > 1 else 0
    val_blocks = blocks[:n_val]
    train_blocks = blocks[n_val:]

    train_ds = Dataset.from_dict({"input_ids": train_blocks})
    val_ds = Dataset.from_dict({"input_ids": val_blocks}) if val_blocks else None

    stats = CorpusStats(
        num_documents=len(documents), num_tokens=len(token_ids), num_blocks=len(blocks)
    )
    return train_ds, val_ds, stats