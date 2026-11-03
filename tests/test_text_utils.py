"""Unit tests for the BM25 tokenizer and company-name suggestions."""
from text_utils import suggest_companies, tokenize

COMPANIES = ["ACRES Commercial Realty Corp.", "RWE AG", "Crombie REIT", "Downer EDI Limited"]


def test_tokenize_drops_numbers_and_stopwords():
    tokens = tokenize("What was the net income in 2022?")
    assert "2022" not in tokens
    assert "what" not in tokens and "the" not in tokens
    assert "net" in tokens and "income" in tokens


def test_tokenize_adds_bigrams():
    assert "net_income" in tokenize("Net income for the year")


def test_partial_name_suggests_full_company():
    assert suggest_companies("ACRES", COMPANIES) == ["ACRES Commercial Realty Corp."]


def test_partial_name_is_case_insensitive():
    assert suggest_companies("acres", COMPANIES) == ["ACRES Commercial Realty Corp."]


def test_typo_falls_back_to_fuzzy_match():
    assert "ACRES Commercial Realty Corp." in suggest_companies("acres comercial", COMPANIES)
