# streamlit_app.py - the DevResolve web app
# run it from the project folder with:  streamlit run app/streamlit_app.py
import json  # to read the documents file
import sys  # to change the import path
from pathlib import Path  # to build file paths
import pandas as pd  # tables and charts
import streamlit as st  # the web app library

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # lets us import the files in the src folder

import config  # our settings
from search import load_store, load_index_info  # loading the index
from answer import answer_question, llm_is_configured, log_query  # the question answering pipeline

st.set_page_config(page_title="DevResolve", layout="wide")  # browser tab title and wide layout


@st.cache_resource
def get_store():
    # loads the index once and keeps it in memory between button clicks
    return load_store(config.INDEX_DIR)


index_ready = (config.INDEX_DIR / "embeddings.npy").exists()  # True once build_index.py has been run
store = get_store() if index_ready else None  # the search store, or None if there is no index yet

# ---------------- sidebar ----------------
st.sidebar.title("DevResolve")
st.sidebar.write("Ask about a Python error or a scikit-learn problem. Answers are based only on the indexed official documentation and cite their sources.")
domain_label = st.sidebar.radio("Documentation to search", ["Both", "Python", "scikit-learn"])  # which docs to use
domain_map = {"Both": "both", "Python": "python", "scikit-learn": "scikit-learn"}  # label -> value used by the search
mode_options = ["Hybrid (meaning + keywords)", "Vector only (meaning)"]  # retrieval choices
default_mode_index = 0 if config.SEARCH_MODE == "hybrid" else 1  # start with the mode from config.py
mode_label = st.sidebar.radio("Retrieval method", mode_options, index=default_mode_index)  # let the user switch
mode = "hybrid" if mode_label.startswith("Hybrid") else "vector"  # value used by the search
top_k = st.sidebar.slider("Passages to retrieve", 3, 8, config.TOP_K)  # how many chunks to use
if llm_is_configured():  # an API key and model are set
    st.sidebar.success("LLM connected: " + config.LLM_MODEL)
else:  # no LLM
    st.sidebar.info("No LLM configured. The app will show retrieved passages only. Add your key to the .env file to get written answers.")

tab_ask, tab_eval, tab_corpus = st.tabs(["Ask", "Evaluation", "Corpus and usage"])  # the three sections

# ---------------- tab 1: ask ----------------
with tab_ask:
    st.header("Ask a question")
    if store is None:  # no index yet
        st.error("The search index does not exist yet. Run `python src/build_index.py` first.")
    else:
        examples = {
            "(write your own)": "",
            "Python: ModuleNotFoundError": "Traceback (most recent call last):\n  File \"app.py\", line 1, in <module>\n    import sklearn\nModuleNotFoundError: No module named 'sklearn'",
            "Python: default argument keeps old values": "Why does a function with a default argument like def f(x, items=[]) keep old values between calls?",
            "scikit-learn: data leakage with scaling": "How do I avoid data leakage when scaling features before cross-validation?",
            "scikit-learn: NotFittedError": "sklearn.exceptions.NotFittedError: This StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
            "scikit-learn: imbalanced metrics": "Which metrics should I use for an imbalanced classification problem?",
            "Out of scope: pandas": "How do I fix SettingWithCopyWarning in pandas?",
        }  # ready-made questions to try
        choice = st.selectbox("Try an example", list(examples.keys()))  # example picker
        question = st.text_area("Your question or traceback", value=examples[choice], height=170)  # the input box
        ask_clicked = st.button("Get answer", type="primary")  # the button

        if ask_clicked:  # the user pressed the button
            if question.strip() == "":  # nothing typed
                st.warning("Type a question or paste a traceback first.")
            else:
                try:
                    with st.spinner("Searching the documentation..."):  # show a spinner while we work
                        output = answer_question(question, store, mode, top_k, domain_map[domain_label])  # run the pipeline
                    log_query(question, domain_map[domain_label], mode, output)  # save one line in the query log
                except Exception as error:  # for example a wrong API key
                    output = None  # nothing to show
                    st.error("Something went wrong: " + str(error))

                if output is not None:  # we have a result
                    if output["abstained"]:  # the system refused to answer
                        st.warning("Not enough evidence in the indexed documentation to answer this reliably.")
                        st.write(output["answer"])
                    else:
                        st.success("Answer based on " + str(len(output["results"])) + " retrieved passages")
                        st.markdown(output["answer"])

                    c1, c2, c3, c4 = st.columns(4)  # four small number boxes
                    c1.metric("Top similarity", "{:.2f}".format(output["top_vector_score"]))  # best cosine score
                    c2.metric("Retrieval time", "{:.0f} ms".format(output["retrieval_ms"]))  # search time
                    if output["llm_used"]:  # the LLM was called
                        c3.metric("LLM time", "{:.1f} s".format(output["llm_seconds"]))
                    else:
                        c3.metric("LLM time", "not used")
                    if output["citation_ok"] is None:  # no answer was checked
                        c4.metric("Citations", "n/a")
                    elif output["citation_ok"]:  # every citation points to a real passage
                        c4.metric("Citations", str(len(output["cited_ids"])) + " valid")
                    else:  # a citation is missing or invalid
                        c4.metric("Citations", "check needed")

                    st.subheader("Retrieved passages")
                    score_rows = []  # data for the small chart
                    for number, chunk in enumerate(output["results"], start=1):  # every passage
                        score_rows.append({"passage": "[" + str(number) + "]", "similarity": round(chunk["vector_score"], 3)})
                    if len(score_rows) > 0:  # there is something to draw
                        st.caption("Similarity of each retrieved passage to your question (higher is closer). If the best passage scores below " + str(config.ABSTAIN_THRESHOLD) + ", the app refuses to answer.")
                        st.bar_chart(pd.DataFrame(score_rows), x="passage", y="similarity")  # one bar per passage
                    for number, chunk in enumerate(output["results"], start=1):  # list the passages
                        cited_note = "  (cited in the answer)" if number in output["cited_ids"] else ""  # mark cited passages
                        label = "[" + str(number) + "] " + chunk["title"] + " > " + chunk["section"] + cited_note
                        with st.expander(label):  # collapsed by default
                            st.markdown("Source: [" + chunk["section_url"] + "](" + chunk["section_url"] + ")  \n" + chunk["domain"] + ", version " + chunk["version"] + ", fetched " + chunk["fetched_at"] + ", similarity " + "{:.2f}".format(chunk["vector_score"]))
                            st.text(chunk["text"])  # the passage text

# ---------------- tab 2: evaluation ----------------
with tab_eval:
    st.header("How well does retrieval work?")
    results_file = config.EXPERIMENTS_DIR / "results.csv"  # written by run_eval.py
    if not results_file.exists():  # nothing measured yet
        st.info("No results yet. Run `python src/run_eval.py` to measure retrieval on the test questions.")
    else:
        results = pd.read_csv(results_file)  # all logged experiments
        results = results.sort_values("run_at").drop_duplicates("config_name", keep="last")  # keep the newest run of each setting
        n_answerable = int(results["n_answerable"].iloc[0])  # answerable test questions
        n_unanswerable = int(results["n_unanswerable"].iloc[0])  # questions that should be refused
        e1, e2, e3 = st.columns(3)  # three number boxes
        e1.metric("Test questions", n_answerable + n_unanswerable)
        e2.metric("Answerable", n_answerable)
        e3.metric("Should be refused", n_unanswerable)

        best = results.sort_values("recall_at_5", ascending=False).iloc[0]  # best setting by recall@5
        st.write("Best recall@5: **{:.0%}** with `{}` (rough 95% range {:.0%} to {:.0%}).".format(best["recall_at_5"], best["config_name"], best["recall_at_5_ci_low"], best["recall_at_5_ci_high"]))
        st.caption("Recall@5 = share of answerable questions where a correct passage is among the top 5 results. With " + str(n_answerable) + " questions, one question changes recall by about " + "{:.1f}".format(100.0 / n_answerable) + " points, so small differences between settings are not conclusive.")

        left, right = st.columns(2)  # two charts side by side
        with left:
            st.subheader("Recall@5 by setting")
            st.bar_chart(results[["config_name", "recall_at_5"]], x="config_name", y="recall_at_5")
        with right:
            st.subheader("MRR@10 by setting")
            st.bar_chart(results[["config_name", "mrr_at_10"]], x="config_name", y="mrr_at_10")
        st.dataframe(results[["config_name", "n_chunks", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr_at_10", "mean_retrieval_ms"]], hide_index=True)

        type_file = config.EXPERIMENTS_DIR / "results_by_type.csv"  # recall by question type
        if type_file.exists():
            by_type = pd.read_csv(type_file)  # read it
            sizes = list(by_type["chunk_size"].unique())  # chunk sizes in the file
            size_to_show = config.CHUNK_SIZE if config.CHUNK_SIZE in sizes else sizes[0]  # use the configured size if possible
            subset = by_type[by_type["chunk_size"] == size_to_show]  # rows of that chunk size
            pivot = subset.pivot(index="question_type", columns="mode", values="recall_at_5")  # one column per method
            if "vector" in pivot.columns and "hybrid" in pivot.columns:  # both methods were measured
                st.subheader("Where does hybrid search help? (chunk size " + str(size_to_show) + ")")
                pivot["hybrid_minus_vector"] = pivot["hybrid"] - pivot["vector"]  # gain from adding keywords
                st.caption("Recall@5 of hybrid minus recall@5 of vector, per question type. Positive = keywords help.")
                st.bar_chart(pivot["hybrid_minus_vector"])  # one bar per question type
                st.dataframe(pivot.round(3))

        sweep_file = config.EXPERIMENTS_DIR / "threshold_sweep.csv"  # abstain threshold experiment
        if sweep_file.exists():
            sweep = pd.read_csv(sweep_file)  # read it
            sweep = sweep[sweep["chunk_size"] == config.CHUNK_SIZE]  # only the configured chunk size
            if len(sweep) > 0:
                st.subheader("Choosing the abstain threshold")
                st.caption("A good threshold refuses most unanswerable questions (correct_abstain_rate high) without refusing answerable ones (false_abstain_rate low). Current setting: " + str(config.ABSTAIN_THRESHOLD))
                st.line_chart(sweep.set_index("threshold")[["false_abstain_rate", "correct_abstain_rate"]])

        summary_file = config.EXPERIMENTS_DIR / "answer_summary.csv"  # answer quality after manual grading
        if summary_file.exists():
            st.subheader("Answer quality (manual grading on a sample)")
            st.dataframe(pd.read_csv(summary_file), hide_index=True)

# ---------------- tab 3: corpus and usage ----------------
with tab_corpus:
    st.header("Corpus and usage")
    info = load_index_info(config.INDEX_DIR)  # facts about the built index
    if info is None:
        st.info("Build the index first (`python src/build_index.py`) to see corpus facts.")
    else:
        k1, k2, k3, k4 = st.columns(4)  # four number boxes
        k1.metric("Pages", info["n_pages"])
        k2.metric("Chunks", info["n_chunks"])
        k3.metric("Chunk size", str(info["chunk_size"]) + " / " + str(info["chunk_overlap"]))
        k4.metric("Built", info["built_at"])
        st.caption("Embedding model: " + info["embedding_model"] + " | vector backend: " + info["vector_backend"] + " | documentation versions: " + str(info["doc_versions"]))
        if store is not None:
            counts = {}  # chunks per page
            for chunk in store["chunks"]:  # every chunk
                key = chunk["domain"] + " | " + chunk["title"]  # page name
                counts[key] = counts.get(key, 0) + 1  # count it
            page_table = pd.DataFrame({"page": list(counts.keys()), "chunks": list(counts.values())})  # table of pages
            page_table = page_table.sort_values("chunks", ascending=False)  # biggest first
            domain_counts = {}  # chunks per domain
            for chunk in store["chunks"]:  # every chunk
                domain_counts[chunk["domain"]] = domain_counts.get(chunk["domain"], 0) + 1
            st.subheader("Chunks per documentation set")
            st.bar_chart(pd.DataFrame({"domain": list(domain_counts.keys()), "chunks": list(domain_counts.values())}), x="domain", y="chunks")
            st.subheader("Largest pages (most chunks)")
            st.dataframe(page_table.head(15), hide_index=True)

    st.subheader("Usage log")
    log_file = config.LOGS_DIR / "queries.csv"  # written by the app after every question
    if not log_file.exists():
        st.info("No questions have been asked yet. Ask something in the first tab and come back.")
    else:
        log = pd.read_csv(log_file)  # read the log
        log["abstained"] = log["abstained"].astype(str) == "True"  # make it True/False
        u1, u2, u3, u4 = st.columns(4)  # four number boxes
        u1.metric("Questions asked", len(log))
        u2.metric("Refused (abstained)", "{:.0%}".format(log["abstained"].mean()))
        u3.metric("Median retrieval time", "{:.0f} ms".format(log["retrieval_ms"].median()))
        u4.metric("Mean LLM time", "{:.1f} s".format(log[log["llm_seconds"] > 0]["llm_seconds"].mean()) if (log["llm_seconds"] > 0).any() else "n/a")
        st.caption("Top similarity score of each question, in the order asked. A sudden drop can mean users ask about topics the documentation does not cover.")
        st.line_chart(log["top_vector_score"].reset_index(drop=True))
        st.dataframe(log.tail(10).iloc[::-1], hide_index=True)  # the 10 newest questions, newest first
        st.caption("The log keeps only the first 80 characters of each question (set LOG_QUESTION_TEXT = False in src/config.py to keep none).")
