"""Unit tests for cleaning and chunking (no models, no API calls, no data files needed)."""
from ingest import MAX_CHARS, chunk_page, clean, split_big_block


def test_clean_removes_image_links_and_html_breaks():
    text = "Intro ![](_page_0_Picture_0.jpeg) text<br>more"
    assert clean(text) == "Intro text more"


def test_clean_unescapes_dollar_signs():
    assert clean(r"Net income of \$40.4 million") == "Net income of $40.4 million"


def test_clean_squeezes_table_padding():
    assert clean("| Revenue      |     100 |") == "| Revenue | 100 |"


def test_short_page_stays_one_chunk():
    page = "First paragraph.\n\nSecond paragraph."
    assert chunk_page(page) == [page]


def test_long_text_is_split_below_limit():
    page = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(10))
    chunks = chunk_page(page)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHARS for c in chunks)


def test_big_table_is_split_with_header_repeated():
    header = "| Item | 2022 | 2021 |\n|---|---|---|"
    rows = [f"| Row {i} with some description text | {i},000 | {i},500 |" for i in range(60)]
    table = header + "\n" + "\n".join(rows)
    pieces = split_big_block(table)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith(header)          # every piece keeps the column headers
        assert len(piece) <= MAX_CHARS


def test_no_table_row_is_lost_when_splitting():
    header = "| Item | Value |\n|---|---|"
    rows = [f"| Row {i} padded with extra words to make it longer | {i} |" for i in range(80)]
    pieces = split_big_block(header + "\n" + "\n".join(rows))
    kept = [line for p in pieces for line in p.splitlines()[2:]]
    assert kept == rows
