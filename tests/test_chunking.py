# test_chunking.py - checks that chunks have the right size, overlap and metadata
from chunk_docs import make_chunks, split_long_block


def make_document():
    # a tiny fake document with two sections
    long_paragraph = ("word " * 300).strip()  # 1499 characters, longer than one chunk
    return {
        "url": "https://example.com/page.html",
        "title": "Example page",
        "domain": "python",
        "version": "1.0",
        "fetched_at": "2026-01-01",
        "sections": [
            {"heading": "Intro", "anchor": "intro", "blocks": ["Short paragraph about errors.", long_paragraph]},
            {"heading": "Second", "anchor": "", "blocks": ["Another paragraph that is long enough to keep."]},
        ],
    }


def test_split_long_block_respects_size():
    pieces = split_long_block("word " * 300, 400)  # cut 1500 characters into pieces of at most 400
    assert len(pieces) >= 4  # needs several pieces
    for piece in pieces:  # every piece is small enough
        assert len(piece) <= 400


def test_chunks_have_metadata_and_size_limit():
    chunks = make_chunks([make_document()], 400, 80)  # small chunk size on purpose
    assert len(chunks) >= 3  # the long paragraph must be split
    for chunk in chunks:  # every chunk
        assert chunk["source_url"] == "https://example.com/page.html"  # keeps the source
        assert chunk["domain"] == "python"  # keeps the domain
        assert len(chunk["text"]) <= 400 + 80 + 1  # soft limit: size plus overlap
        assert chunk["embed_text"].startswith("Example page | ")  # the header is added for embedding
    assert chunks[0]["section_url"].endswith("#intro")  # section link uses the anchor
    ids = [chunk["chunk_id"] for chunk in chunks]  # chunk ids
    assert len(ids) == len(set(ids))  # all ids are different


def test_zero_overlap_does_not_crash():
    chunks = make_chunks([make_document()], 400, 0)  # no overlap
    assert len(chunks) >= 3  # still produces chunks
    total = sum(len(chunk["text"]) for chunk in chunks)  # all text together
    assert total < 2200  # nothing was duplicated by mistake (the text has about 1600 characters)


def test_every_exception_definition_gets_its_own_chunk():
    document = {
        "url": "https://example.com/exceptions.html",
        "title": "Built-in Exceptions",
        "domain": "python",
        "version": "1.0",
        "fetched_at": "2026-01-01",
        "sections": [{"heading": "Concrete exceptions", "anchor": "", "blocks": [
            "This introduction paragraph is long enough to be kept as a chunk.",
            "exception AlphaError",
            "Raised when alpha happens somewhere in a program.",
            "exception BetaError",
            "Raised when beta happens somewhere in a program.",
        ]}],
    }
    chunks = make_chunks([document], 800, 150)
    assert len(chunks) == 3  # introduction, AlphaError, BetaError
    assert "AlphaError" in chunks[1]["text"] and "BetaError" not in chunks[1]["text"]  # no mixing of two exceptions
    assert "BetaError" in chunks[2]["text"] and "AlphaError" not in chunks[2]["text"]
