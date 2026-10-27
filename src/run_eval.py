# run_eval.py - measures retrieval quality for different settings (experiments E0, E1, E2)
#   E0 = baseline: chunk size 800 + vector search
#   E1 = other chunk sizes (500 and 1200) with vector search
#   E2 = hybrid search (vector + BM25) for every chunk size
# run it with:  python src/run_eval.py
import hashlib  # to detect when cached embeddings are out of date
import json  # to read the questions
from datetime import datetime  # to timestamp the results
import numpy as np  # arrays
import pandas as pd  # tables and CSV files
import config  # our settings
import search  # embedding and searching
from chunk_docs import load_documents, make_chunks  # reading pages and cutting chunks
from metrics import first_hit_rank, recall_at_k, mean_reciprocal_rank, bootstrap_ci  # the metrics

CHUNK_SETTINGS = [(500, 100), (800, 150), (1200, 240)]  # (chunk size, overlap) pairs to compare
MODES = ["vector", "hybrid"]  # search methods to compare
TOP_K = 10  # we retrieve 10 chunks per question and compute recall@1, 3, 5 and 10 from them
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]  # abstain thresholds to try

config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)  # create the experiments folder
(config.EXPERIMENTS_DIR / "details").mkdir(parents=True, exist_ok=True)  # folder for per-question results

documents = load_documents()  # read the parsed pages
questions = []  # the test questions
with open(config.QUESTIONS_FILE, encoding="utf-8") as f:  # open the question file
    for line in f:  # one JSON object per line
        if line.strip() != "":  # skip empty lines
            questions.append(json.loads(line))  # convert text to a dictionary
n_answerable = 0  # count answerable questions
for q in questions:  # every question
    if q["is_answerable"]:  # it can be answered from the corpus
        n_answerable = n_answerable + 1
n_unanswerable = len(questions) - n_answerable  # the rest should be refused
print("Questions:", len(questions), "| answerable:", n_answerable, "| unanswerable:", n_unanswerable)
print("Vector backend:", "FAISS" if search.faiss is not None else "numpy (FAISS not installed)")

result_rows = []  # one row per experiment (setting)
type_rows = []  # recall by question type
sweep_rows = []  # abstain threshold sweep
run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # time stamp for this run

for chunk_size, chunk_overlap in CHUNK_SETTINGS:  # every chunk setting
    chunks = make_chunks(documents, chunk_size, chunk_overlap)  # cut the pages into chunks
    texts = [chunk["embed_text"] for chunk in chunks]  # texts to embed
    print("")  # empty line
    print("Chunk size", chunk_size, "overlap", chunk_overlap, "->", len(chunks), "chunks")  # progress

    cache_dir = config.EXPERIMENT_INDEX_DIR / ("chunk" + str(chunk_size) + "_overlap" + str(chunk_overlap))  # cache folder
    fingerprint = hashlib.md5((config.EMBEDDING_MODEL + "".join(texts)).encode("utf-8")).hexdigest()  # identifies these exact texts
    fingerprint_file = cache_dir / "fingerprint.txt"  # where the fingerprint is stored
    embeddings_file = cache_dir / "embeddings.npy"  # where the vectors are stored
    if fingerprint_file.exists() and embeddings_file.exists() and fingerprint_file.read_text() == fingerprint:  # cache is valid
        embeddings = np.load(embeddings_file)  # reuse the saved vectors
        print("Loaded cached embeddings")  # progress
    else:  # no valid cache
        embeddings = search.embed_texts(texts, show_progress=True)  # embed all chunks
        cache_dir.mkdir(parents=True, exist_ok=True)  # create the cache folder
        np.save(embeddings_file, embeddings)  # save the vectors
        fingerprint_file.write_text(fingerprint)  # save the fingerprint

    store = search.make_store(chunks, embeddings)  # build the search store

    for mode in MODES:  # vector and hybrid
        config_name = "chunk" + str(chunk_size) + "_" + mode  # name of this experiment
        detail_rows = []  # one row per question
        answerable_ranks = []  # hit rank of every answerable question
        retrieval_times = []  # retrieval time of every question

        for q in questions:  # run every question
            output = search.search(q["question"], store, mode=mode, top_k=TOP_K, domain="both")  # retrieve chunks
            retrieved = output["results"]  # the chunks
            if q["is_answerable"]:  # we can measure a hit
                rank = first_hit_rank(retrieved, q)  # position of the first correct chunk (or None)
                answerable_ranks.append(rank)  # remember it
            else:
                rank = None  # no gold chunk exists
            retrieval_times.append(output["retrieval_ms"])  # remember the time
            urls = []  # top 5 page addresses
            for chunk in retrieved[:5]:  # first five chunks
                urls.append(chunk["source_url"].replace("https://", ""))  # shorter address
            detail_rows.append({
                "question_id": q["question_id"],
                "question_type": q["question_type"],
                "is_answerable": q["is_answerable"],
                "question": q["question"][:120].replace("\n", " "),
                "hit_rank": rank,
                "hit_at_1": 1 if (rank is not None and rank <= 1) else 0,
                "hit_at_3": 1 if (rank is not None and rank <= 3) else 0,
                "hit_at_5": 1 if (rank is not None and rank <= 5) else 0,
                "top_vector_score": round(output["top_vector_score"], 4),
                "top5_urls": " | ".join(urls),
            })  # save the details of this question

        details = pd.DataFrame(detail_rows)  # table of all questions
        details.to_csv(config.EXPERIMENTS_DIR / "details" / (config_name + ".csv"), index=False)  # save for inspection
        answerable_table = details[details["is_answerable"] == True]  # rows of answerable questions
        unanswerable_table = details[details["is_answerable"] == False]  # rows of unanswerable questions

        low, high = bootstrap_ci(list(answerable_table["hit_at_5"]))  # rough 95% range for recall@5

        threshold = config.ABSTAIN_THRESHOLD  # the threshold from config.py
        false_abstain = float((answerable_table["top_vector_score"] < threshold).mean())  # answerable questions we wrongly refuse
        if len(unanswerable_table) > 0:  # there are unanswerable questions
            correct_abstain = float((unanswerable_table["top_vector_score"] < threshold).mean())  # unanswerable questions we correctly refuse
        else:
            correct_abstain = 0.0
        right_decisions = (1 - false_abstain) * len(answerable_table) + correct_abstain * len(unanswerable_table)  # decisions that were right
        abstention_accuracy = right_decisions / len(details)  # share of right decisions

        result_rows.append({
            "run_at": run_time,
            "config_name": config_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "mode": mode,
            "embedding_model": config.EMBEDDING_MODEL,
            "n_chunks": len(chunks),
            "n_answerable": len(answerable_table),
            "n_unanswerable": len(unanswerable_table),
            "recall_at_1": round(recall_at_k(answerable_ranks, 1), 3),
            "recall_at_3": round(recall_at_k(answerable_ranks, 3), 3),
            "recall_at_5": round(recall_at_k(answerable_ranks, 5), 3),
            "recall_at_10": round(recall_at_k(answerable_ranks, 10), 3),
            "recall_at_5_ci_low": round(low, 3),
            "recall_at_5_ci_high": round(high, 3),
            "mrr_at_10": round(mean_reciprocal_rank(answerable_ranks), 3),
            "mean_retrieval_ms": round(float(np.mean(retrieval_times)), 1),
            "abstain_threshold": threshold,
            "abstention_accuracy": round(abstention_accuracy, 3),
            "false_abstain_rate": round(false_abstain, 3),
            "correct_abstain_rate": round(correct_abstain, 3),
        })  # save the numbers of this experiment

        for question_type, group in answerable_table.groupby("question_type"):  # recall@5 for every question type
            type_rows.append({
                "config_name": config_name,
                "chunk_size": chunk_size,
                "mode": mode,
                "question_type": question_type,
                "n": len(group),
                "recall_at_5": round(float(group["hit_at_5"].mean()), 3),
            })

        if mode == "vector":  # the threshold sweep only depends on vector scores, so we do it once per chunk size
            for t in THRESHOLDS:  # try every threshold
                sweep_rows.append({
                    "chunk_size": chunk_size,
                    "threshold": t,
                    "false_abstain_rate": round(float((answerable_table["top_vector_score"] < t).mean()), 3),
                    "correct_abstain_rate": round(float((unanswerable_table["top_vector_score"] < t).mean()), 3) if len(unanswerable_table) > 0 else 0.0,
                })

        print("  ", config_name, "recall@5 =", round(recall_at_k(answerable_ranks, 5), 3))  # quick progress line

# ---------- save the results ----------
results = pd.DataFrame(result_rows)  # table of all experiments
results_file = config.EXPERIMENTS_DIR / "results.csv"  # the experiment log (new runs are added at the bottom)
results.to_csv(results_file, mode="a", header=not results_file.exists(), index=False)  # append to the log
pd.DataFrame(type_rows).to_csv(config.EXPERIMENTS_DIR / "results_by_type.csv", index=False)  # latest run only
pd.DataFrame(sweep_rows).to_csv(config.EXPERIMENTS_DIR / "threshold_sweep.csv", index=False)  # latest run only

# ---------- print tables you can copy and send back ----------
pd.set_option("display.width", 220)  # wider printing
pd.set_option("display.max_columns", 30)  # show all columns
print("")
print("===== RESULTS (retrieval, answerable questions only for recall/MRR) =====")
show_columns = ["config_name", "n_chunks", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_5_ci_low", "recall_at_5_ci_high", "recall_at_10", "mrr_at_10", "mean_retrieval_ms", "abstention_accuracy", "false_abstain_rate", "correct_abstain_rate"]
print(results[show_columns].to_string(index=False))  # main table
print("")
print("===== RECALL@5 BY QUESTION TYPE =====")
by_type = pd.DataFrame(type_rows)  # table by question type
print(by_type.pivot(index="question_type", columns="config_name", values="recall_at_5").to_string())  # one column per experiment
print("")
print("===== ABSTAIN THRESHOLD SWEEP (chunk size", config.CHUNK_SIZE, ") =====")
sweep = pd.DataFrame(sweep_rows)  # sweep table
print(sweep[sweep["chunk_size"] == config.CHUNK_SIZE].to_string(index=False))  # sweep for the configured chunk size
print("")
print("Saved: experiments/results.csv, results_by_type.csv, threshold_sweep.csv and experiments/details/*.csv")
