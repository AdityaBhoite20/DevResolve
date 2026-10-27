# test_metrics.py - checks the retrieval metrics on small examples
from metrics import is_hit, first_hit_rank, recall_at_k, mean_reciprocal_rank, bootstrap_ci


def make_chunk(url, text):
    # a minimal fake chunk
    return {"source_url": url, "section": "Heading", "text": text}


def test_is_hit_needs_right_page_and_keyword():
    question = {"gold_urls": ["a"], "gold_keywords": ["KeyError"]}  # gold page "a" and keyword "KeyError"
    assert is_hit(make_chunk("a", "A keyerror happens"), question)  # right page, keyword (any case)
    assert not is_hit(make_chunk("a", "something else"), question)  # right page, no keyword
    assert not is_hit(make_chunk("b", "KeyError"), question)  # wrong page


def test_first_hit_rank_and_recall():
    question = {"gold_urls": ["a"], "gold_keywords": []}  # no keyword rule
    results = [make_chunk("x", "t"), make_chunk("a", "t"), make_chunk("y", "t")]  # gold page is second
    assert first_hit_rank(results, question) == 2
    assert first_hit_rank([make_chunk("x", "t")], question) is None  # no hit at all
    ranks = [1, 2, None, 7]  # four questions
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.5
    assert recall_at_k(ranks, 10) == 0.75


def test_mrr_and_ci():
    assert mean_reciprocal_rank([1, 2, None, None]) == (1.0 + 0.5) / 4  # (1/1 + 1/2 + 0 + 0) / 4
    low, high = bootstrap_ci([1, 1, 0, 1, 0, 1, 1, 1, 0, 1])  # average is 0.7
    assert low <= 0.7 <= high  # the range contains the average
    assert 0.0 <= low and high <= 1.0  # stays between 0 and 1
