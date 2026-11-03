"""Small, pure text helpers with no heavy dependencies, so they can be unit-tested in CI."""
import difflib
import re

STOPWORDS = {"what", "was", "were", "the", "for", "of", "a", "an", "is", "in", "to",
             "and", "did", "does", "do", "how", "much", "many", "year", "company"}


def tokenize(text):
    """BM25 tokens: lowercase letters only (numbers dropped), no stopwords, plus bigrams."""
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def suggest_companies(name, companies, n=3):
    """Suggestions for an unknown company name: partial matches first, then typo matches.
    Case-insensitive in both steps."""
    query = name.strip().lower()
    by_lower = {c.lower(): c for c in companies}
    partial = [c for c in companies if query in c.lower()][:n]
    if partial:
        return partial
    matches = difflib.get_close_matches(query, list(by_lower), n=n, cutoff=0.3)
    return [by_lower[m] for m in matches]
