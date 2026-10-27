# bm25.py - a small BM25 keyword search written from scratch (no extra library needed)
import re  # to split text into words
import numpy as np  # for the score array


def tokenize(text):
    # lowercase the text and cut it into words
    # underscores stay inside a word, so "fit_transform" and "ModuleNotFoundError" stay as one word each
    return re.findall(r"\w+", text.lower())


def build_bm25(token_lists):
    # token_lists is a list that holds one list of words for every chunk
    n_docs = len(token_lists)  # number of chunks
    doc_lengths = np.array([len(tokens) for tokens in token_lists], dtype="float64")  # words in each chunk
    if n_docs > 0:  # avoid dividing by zero for an empty corpus
        avg_length = float(doc_lengths.mean())  # average chunk length in words
    else:
        avg_length = 1.0  # any number works for an empty corpus
    postings = {}  # word -> {chunk number: how many times the word appears in that chunk}
    for doc_id, tokens in enumerate(token_lists):  # go through every chunk
        for token in tokens:  # go through every word in the chunk
            if token not in postings:  # first time we see this word
                postings[token] = {}  # create an entry for it
            postings[token][doc_id] = postings[token].get(doc_id, 0) + 1  # count the word in this chunk
    return {"postings": postings, "doc_lengths": doc_lengths, "avg_length": avg_length, "n_docs": n_docs}


def bm25_scores(bm25, query_tokens, k1=1.5, b=0.75):
    scores = np.zeros(bm25["n_docs"])  # one score per chunk, all start at 0
    for token in set(query_tokens):  # each different query word once
        if token not in bm25["postings"]:  # the word never appears in the corpus
            continue  # nothing to add
        docs_with_word = bm25["postings"][token]  # chunks that contain the word
        doc_freq = len(docs_with_word)  # in how many chunks the word appears
        idf = np.log(1 + (bm25["n_docs"] - doc_freq + 0.5) / (doc_freq + 0.5))  # rare words get a bigger weight
        for doc_id, count in docs_with_word.items():  # score every chunk that contains the word
            length_part = 1 - b + b * bm25["doc_lengths"][doc_id] / bm25["avg_length"]  # long chunks are penalised
            scores[doc_id] += idf * count * (k1 + 1) / (count + k1 * length_part)  # the BM25 formula
    return scores
