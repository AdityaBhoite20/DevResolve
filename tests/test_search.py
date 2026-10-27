# test_search.py - checks BM25, query preparation and the search function with a fake embedding function
import zlib  # a stable hash for the fake embeddings
import numpy as np
import search
from bm25 import tokenize, build_bm25, bm25_scores


def fake_embed(texts, show_progress=False):
    # fake embeddings: count hashed words into 64 buckets (similar words -> similar vectors)
    vectors = np.zeros((len(texts), 64), dtype="float32")
    for i, text in enumerate(texts):
        for word in tokenize(text):
            vectors[i, zlib.crc32(word.encode("utf-8")) % 64] += 1.0
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] = vectors[i] / norm
    return vectors


def make_test_store():
    topics = [
        ("python", "ModuleNotFoundError is raised when a module could not be located by import"),
        ("python", "A list is mutable but a tuple is immutable sequence of items"),
        ("scikit-learn", "StandardScaler removes the mean and scales every feature to unit variance"),
        ("scikit-learn", "GridSearchCV tries every combination of the parameter grid with cross validation"),
    ]
    chunks = []
    for number, (domain, text) in enumerate(topics):
        chunks.append({"chunk_id": str(number), "source_url": "u" + str(number), "section_url": "u" + str(number),
                       "title": "T" + str(number), "domain": domain, "section": "S", "version": "1",
                       "fetched_at": "2026-01-01", "text": text, "embed_text": text})
    embeddings = fake_embed([chunk["embed_text"] for chunk in chunks])
    return search.make_store(chunks, embeddings)


def test_bm25_prefers_rare_exact_word():
    token_lists = [tokenize("fit_transform learns and applies"), tokenize("a general text about models"), tokenize("another general text")]
    bm25 = build_bm25(token_lists)
    scores = bm25_scores(bm25, tokenize("fit_transform"))
    assert scores[0] > 0 and scores[1] == 0 and scores[2] == 0  # only the first chunk contains the word


def test_prepare_query_uses_last_traceback_line():
    text = "Why?\nTraceback (most recent call last):\n  File \"a.py\", line 1\nKeyError: 'apple'\n"
    prepared = search.prepare_query(text)
    assert prepared.endswith("KeyError: 'apple'")  # the error line is used
    assert "File" not in prepared  # the noisy middle lines are dropped


def test_search_modes_and_domain_filter():
    search.embed_texts = fake_embed  # use the fake embeddings instead of the real model
    store = make_test_store()
    out = search.search("ModuleNotFoundError import", store, mode="vector", top_k=2)
    assert out["results"][0]["chunk_id"] == "0"  # best chunk is the matching one
    out = search.search("GridSearchCV parameter grid", store, mode="hybrid", top_k=2)
    assert out["results"][0]["chunk_id"] == "3"  # hybrid finds it too
    out = search.search("mean variance scaler", store, mode="vector", top_k=4, domain="python")
    for chunk in out["results"]:  # the domain filter removes the other documentation set
        assert chunk["domain"] == "python"
    assert out["top_vector_score"] <= 1.0001  # cosine similarity never exceeds 1


def test_prepare_query_ignores_a_question_written_after_the_traceback():
    text = "My script crashes with this:\nTraceback (most recent call last):\n  File \"app.py\", line 1, in <module>\n    import sklearn\nModuleNotFoundError: No module named 'sklearn'\nWhat does it mean?"
    prepared = search.prepare_query(text)
    assert "ModuleNotFoundError: No module named 'sklearn'" in prepared  # the error line is used
    assert "What does it mean" not in prepared  # the trailing question is not mistaken for the error
    assert prepared.startswith("My script crashes with this:")  # the words before the traceback are kept


def test_prepare_query_handles_chained_errors_and_flat_tracebacks():
    chained = "Traceback (most recent call last):\n  File \"a.py\", line 2\nKeyError: 'a'\n\nDuring handling of the above exception, another exception occurred:\n\nTraceback (most recent call last):\n  File \"a.py\", line 4\nValueError: bad value"
    assert search.prepare_query(chained).endswith("ValueError: bad value")  # the last error wins
    flat = "Traceback (most recent call last):\nFile \"a.py\", line 1, in <module>\nimport x\nImportError: cannot import name 'y'"
    assert search.prepare_query(flat).endswith("ImportError: cannot import name 'y'")  # works even when the indentation was lost
