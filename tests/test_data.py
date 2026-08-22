"""Tests for domain corpus chunking and instruction dataset formatting."""

from __future__ import annotations

import json

import pytest

from src.data.domain_corpus import chunk_token_ids, load_raw_documents
from src.data.instruction_dataset import (
    InstructionExample,
    load_instruction_examples,
)


def test_chunk_token_ids_drops_incomplete_trailing_block() -> None:
    tokens = list(range(25))  # 25 tokens
    blocks = chunk_token_ids(tokens, block_size=10)

    assert len(blocks) == 2
    assert blocks[0] == list(range(0, 10))
    assert blocks[1] == list(range(10, 20))


def test_chunk_token_ids_rejects_non_positive_block_size() -> None:
    with pytest.raises(ValueError):
        chunk_token_ids([1, 2, 3], block_size=0)


def test_load_raw_documents_reads_txt_files(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("Contract clause one.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Contract clause two.", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("not a txt file", encoding="utf-8")

    docs = load_raw_documents(tmp_path)

    assert len(docs) == 2
    assert "Contract clause one." in docs
    assert "Contract clause two." in docs


def test_load_raw_documents_raises_on_missing_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_documents(tmp_path / "does_not_exist")


def test_instruction_example_format_with_and_without_input() -> None:
    with_input = InstructionExample(
        instruction="Summarize the clause.", input="Force majeure text.", output="Summary."
    )
    no_input = InstructionExample(instruction="Define indemnification.", input="", output="Def.")

    assert "### Input:\nForce majeure text." in with_input.format()
    assert "### Input:" not in no_input.format()
    assert no_input.format().endswith("Def.")


def test_load_instruction_examples_parses_jsonl(tmp_path) -> None:
    path = tmp_path / "instructions.jsonl"
    records = [
        {"instruction": "Explain X.", "output": "X is..."},
        {"instruction": "Explain Y.", "input": "context", "output": "Y is..."},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    examples = load_instruction_examples(path)

    assert len(examples) == 2
    assert examples[1].input == "context"


def test_load_instruction_examples_rejects_missing_fields(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"instruction": "Missing output"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_instruction_examples(path)