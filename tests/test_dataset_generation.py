"""
Tests for the fine-tuning dataset generator.

The parser bug these guard against was real: only the single line carrying the
"A:" prefix was kept, so every multi-line answer -- which is most of the useful
ones, because the generation prompt asks for bullet points -- was silently
truncated to its first line.  Fine-tuning on truncated answers teaches a model
to stop mid-thought, which is worse than not fine-tuning at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_dataset import _parse_qa_pairs  # noqa: E402


def test_multiline_answer_is_captured_in_full():
    text = """Q: What hostel amenities are provided?
A: Hostel amenities include:
  - Geysers for hot water
  - RO drinking water
  - Washing machines
"""
    (pair,) = _parse_qa_pairs(text)
    assert "Geysers" in pair["output"]
    assert "Washing machines" in pair["output"]


def test_multiple_pairs_are_separated():
    text = """Q: First question?
A: First answer.

Q: Second question?
A: Second answer.
"""
    pairs = _parse_qa_pairs(text)
    assert [p["instruction"] for p in pairs] == ["First question?", "Second question?"]
    assert pairs[0]["output"] == "First answer."


def test_question_answer_long_form_prefixes_accepted():
    pairs = _parse_qa_pairs("Question: What is the fee?\nAnswer: Rs 17,24,000.")
    assert pairs[0]["instruction"] == "What is the fee?"
    assert pairs[0]["output"] == "Rs 17,24,000."


def test_answer_without_question_is_discarded():
    assert _parse_qa_pairs("A: an orphaned answer with no question") == []


def test_question_without_answer_is_discarded():
    assert _parse_qa_pairs("Q: a question nobody answered") == []


def test_records_use_the_alpaca_field_names_the_notebook_expects():
    (pair,) = _parse_qa_pairs("Q: q?\nA: a.")
    assert set(pair) == {"instruction", "input", "output"}


def test_empty_input_is_handled():
    assert _parse_qa_pairs("") == []
