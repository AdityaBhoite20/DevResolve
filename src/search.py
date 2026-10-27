# search.py - turns text into vectors, builds the search store and finds the best chunks
import json  # to read the chunk files
import re  # to recognise error lines like KeyError: 'x'
import time  # to measure how long a search takes
import numpy as np  # arrays and maths
import config  # our settings
from bm25 import tokenize, build_bm25, bm25_scores  # the keyword search

try:
    import faiss  # optional library for fast vector search
except ImportError:
    faiss = None  # if faiss is not installed we use plain numpy instead (same results)

_model = None  # the embedding model is loaded once and kept here
RRF_K = 60  # constant used when we merge two rankings (60 is the common default)
N_CANDIDATES = 30  # how many candidates each search method gives to the merge


def get_model():
    global _model  # we want to change the variable defined above
    if _model is None:  # only load the model the first time
        from sentence_transformers import SentenceTransformer  # imported here so the tests can run without it
        _model = SentenceTransformer(config.EMBEDDING_MODEL)  # downloads the model the first time (PyTorch runs it)
    return _model


def embed_texts(texts, show_progress=False):
    # turns a list of texts into a numpy array with one vector per text
    model = get_model()  # load (or reuse) the model
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )  # normalize_embeddings=True makes every vector length 1, so dot product = cosine similarity
    return np.ascontiguousarray(vectors, dtype="float32")  # faiss needs float32 numbers


def prepare_query(text):
    # prepares the text we search with
    text = text.strip()  # remove spaces at both ends
    marker = "Traceback (most recent call last)"  # the first line of every Python traceback
    if marker in text:  # the user pasted a traceback
        words_before = text.split(marker)[0].strip()  # what the user wrote before the traceback
        after = text.split(marker, 1)[1].lstrip(":")  # everything after the marker
        error_line = ""  # the line with the error name and message
        last_plain_line = ""  # backup: the last line that is not part of the stack frames
        for line in after.splitlines():  # go through every line after the marker
            if line.strip() == "" or line.startswith(" ") or line.startswith("\t"):  # empty or indented lines are stack frames
                continue
            if line.startswith("Traceback"):  # the start of another traceback (chained errors)
                continue
            last_plain_line = line.strip()  # remember it as a backup
            if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Warning|Interrupt|Exit|Iteration)\b", line.strip()):  # looks like ValueError: ...
                error_line = line.strip()  # keep the last error line we find
        if error_line == "":  # no error name was recognised
            error_line = last_plain_line  # use the backup
        return (words_before[:300] + " " + error_line).strip()  # search with both
    return text[:500]  # very long questions are cut


def make_store(chunks, embeddings):
    # builds everything we need for searching from the chunks and their vectors
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")  # make sure the format is right
    store = {}  # the search store is a dictionary
    store["chunks"] = chunks  # the chunk records
    store["embeddings"] = embeddings  # one vector per chunk
    store["domains"] = np.array([chunk["domain"] for chunk in chunks])  # domain of every chunk (for filtering)
    token_lists = []  # list of word lists, one per chunk
    for chunk in chunks:  # every chunk
        token_lists.append(tokenize(chunk["embed_text"]))  # split its text into words
    store["bm25"] = build_bm25(token_lists)  # keyword index
    if faiss is not None and len(chunks) > 0:  # faiss is available
        index = faiss.IndexFlatIP(embeddings.shape[1])  # exact search with inner product
        index.add(embeddings)  # add all vectors
        store["faiss_index"] = index  # keep the index
    else:
        store["faiss_index"] = None  # numpy will be used
    return store


def load_store(index_dir):
    # loads a saved index folder (made by build_index.py)
    chunks = []  # chunk records
    with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:  # open the chunks file
        for line in f:  # one JSON object per line
            if line.strip() != "":  # skip empty lines
                chunks.append(json.loads(line))  # convert text to dictionary
    embeddings = np.load(index_dir / "embeddings.npy")  # load the vectors
    return make_store(chunks, embeddings)  # build the store


def load_index_info(index_dir):
    # reads index_info.json (settings used to build the index), or returns None
    info_file = index_dir / "index_info.json"  # file path
    if not info_file.exists():  # no file
        return None
    with open(info_file, encoding="utf-8") as f:  # open it
        return json.load(f)  # return the dictionary


def search(query, store, mode="hybrid", top_k=5, domain="both"):
    # finds the best chunks for a query
    # mode "vector" = meaning only, mode "hybrid" = meaning + BM25 keywords merged
    # domain "both", "python" or "scikit-learn"
    start = time.time()  # start the timer
    search_text = prepare_query(query)  # prepare the text
    query_vector = embed_texts([search_text])  # one vector for the query
    n_chunks = len(store["chunks"])  # number of chunks in the store

    if store["faiss_index"] is not None:  # use faiss
        scores, ids = store["faiss_index"].search(query_vector, n_chunks)  # score every chunk, best first
        vector_scores = np.zeros(n_chunks, dtype="float32")  # one score per chunk
        vector_scores[ids[0]] = scores[0]  # put each score at the chunk's position
    else:  # use numpy
        vector_scores = store["embeddings"] @ query_vector[0]  # dot product with every chunk vector

    if domain == "both":  # no filter
        allowed = np.ones(n_chunks, dtype=bool)  # every chunk is allowed
    else:  # only one domain
        allowed = store["domains"] == domain  # True for chunks of that domain
    masked_scores = np.where(allowed, vector_scores, -1.0)  # chunks of other domains get a very low score
    vector_order = []  # best chunks by meaning, best first
    for chunk_index in np.argsort(-masked_scores)[:N_CANDIDATES]:  # look at the best candidates
        if allowed[chunk_index]:  # skip chunks of the other documentation set
            vector_order.append(int(chunk_index))  # keep the chunk number
    if len(vector_order) > 0:  # at least one chunk is allowed
        top_vector_score = float(masked_scores[vector_order[0]])  # best similarity (used to decide abstaining)
    else:
        top_vector_score = -1.0  # nothing to search in

    if mode == "vector":  # meaning only
        final_order = vector_order[:top_k]  # take the best k
    else:  # hybrid
        keyword_scores = bm25_scores(store["bm25"], tokenize(search_text))  # BM25 score of every chunk
        keyword_scores = np.where(allowed, keyword_scores, 0.0)  # remove other domains
        keyword_order = np.argsort(-keyword_scores)[:N_CANDIDATES]  # best keyword matches
        fused = {}  # chunk number -> merged score
        for rank, chunk_index in enumerate(vector_order):  # ranking from vector search
            fused[int(chunk_index)] = fused.get(int(chunk_index), 0.0) + 1.0 / (RRF_K + rank + 1)  # reciprocal rank fusion
        for rank, chunk_index in enumerate(keyword_order):  # ranking from keyword search
            if keyword_scores[chunk_index] > 0:  # only chunks that really match a word
                fused[int(chunk_index)] = fused.get(int(chunk_index), 0.0) + 1.0 / (RRF_K + rank + 1)  # add its share
        final_order = sorted(fused, key=fused.get, reverse=True)[:top_k]  # best merged scores first

    results = []  # chunks we return
    for rank, chunk_index in enumerate(final_order):  # every chosen chunk
        chunk = dict(store["chunks"][int(chunk_index)])  # copy the chunk record
        chunk["vector_score"] = float(vector_scores[int(chunk_index)])  # its similarity to the query
        chunk["rank"] = rank + 1  # position in the result list
        results.append(chunk)  # keep it

    retrieval_ms = (time.time() - start) * 1000  # elapsed time in milliseconds
    return {
        "results": results,
        "top_vector_score": top_vector_score,
        "retrieval_ms": retrieval_ms,
        "search_text": search_text,
    }
