# build_index.py - builds the final search index that the app uses
# run it with:  python src/build_index.py
import hashlib  # to make a fingerprint of the chunk texts
import json  # to write files
from datetime import datetime  # to record when the index was built
import numpy as np  # to save the vectors
import config  # our settings
import search  # we use search.faiss and search.embed_texts
from chunk_docs import load_documents, make_chunks  # reading pages and cutting chunks

documents = load_documents()  # read data/processed/documents.jsonl
chunks = make_chunks(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)  # cut the pages into chunks
print("Pages:", len(documents), "| Chunks:", len(chunks))  # show the size of the corpus
if len(chunks) == 0:  # nothing to index
    raise SystemExit("No chunks found. Run fetch_docs.py and parse_docs.py first.")

texts = [chunk["embed_text"] for chunk in chunks]  # the text we embed for every chunk
print("Embedding with", config.EMBEDDING_MODEL, "(this can take a few minutes on a CPU)...")  # progress message
embeddings = search.embed_texts(texts, show_progress=True)  # turn all chunks into vectors

config.INDEX_DIR.mkdir(parents=True, exist_ok=True)  # create indexes/final if needed
np.save(config.INDEX_DIR / "embeddings.npy", embeddings)  # save the vectors
with open(config.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:  # save the chunk records
    for chunk in chunks:  # one JSON object per line
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")  # write the line

versions = {}  # documentation versions found in the corpus, per domain
for document in documents:  # every page
    domain = document["domain"]  # python or scikit-learn
    if domain not in versions:  # first page of this domain
        versions[domain] = []  # start an empty list
    if document["version"] not in versions[domain]:  # new version
        versions[domain].append(document["version"])  # keep it

info = {
    "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "embedding_model": config.EMBEDDING_MODEL,
    "chunk_size": config.CHUNK_SIZE,
    "chunk_overlap": config.CHUNK_OVERLAP,
    "n_pages": len(documents),
    "n_chunks": len(chunks),
    "doc_versions": versions,
    "vector_backend": "faiss" if search.faiss is not None else "numpy",
    "texts_fingerprint": hashlib.md5("".join(texts).encode("utf-8")).hexdigest(),
}  # information about this index (this is our index version record)
with open(config.INDEX_DIR / "index_info.json", "w", encoding="utf-8") as f:  # save it
    json.dump(info, f, indent=2)  # write it nicely formatted

print("Saved index to:", config.INDEX_DIR)  # where the files are
print("Vector backend:", info["vector_backend"])  # faiss or numpy
print("Documentation versions:", versions)  # what we indexed
