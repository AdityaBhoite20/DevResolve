# DevResolve

A small RAG (retrieval-augmented generation) assistant for **Python errors** and **scikit-learn questions**.
It searches a local copy of the official Python and scikit-learn documentation, gives the best passages to an LLM,
and returns a short answer with numbered citations. If the documentation does not contain the answer, it says so.

The main point of the project is **measurement**: a fixed test set of questions is used to compare chunk sizes and
vector search against hybrid (vector + BM25) search, and to tune when the system should refuse to answer.

```
question -> prepare query -> embed (Sentence-Transformers / PyTorch) -> vector search (+ BM25 keywords)
         -> merge rankings (RRF) -> top passages -> LLM answer with [1] [2] citations -> citation check
                                  \-> best score too low? -> "insufficient evidence"
```

## What you need first

1. Python 3.10-3.12 (`python --version`) and Git.
2. About 3 GB of free disk space and an internet connection (PyTorch and the embedding model are large downloads).
3. Windows PowerShell: if activating the virtual environment says "running scripts is disabled", run once in that window:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
4. An LLM API key (any provider with an OpenAI-compatible chat API, or a local Ollama model). Without a key, retrieval,
   evaluation and the app's retrieved-passages view still work. Written answers need the key.
5. Always run the commands from the project folder (the folder that contains `README.md`).

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install faiss-cpu
copy .env.example .env
```

- `pip install faiss-cpu` is optional. If it fails, skip it: the project falls back to numpy automatically.
- Open `.env` and fill in `LLM_API_KEY`, `LLM_MODEL` (and `LLM_BASE_URL` if you do not use OpenAI). Model names change, so copy the current one from your provider's documentation.

## Run order

```powershell
python src/fetch_docs.py        # 1. download the 51 documentation pages (about 1-2 minutes)
python src/parse_docs.py        # 2. extract clean text sections - READ the summary at the end
python src/chunk_docs.py        # 3. (optional) look at the chunk statistics
python src/run_eval.py          # 4. run experiments E0-E2 on the test questions (several minutes the first time)
```

Look at the tables printed by step 4, then open `src/config.py` and set `CHUNK_SIZE`, `CHUNK_OVERLAP`, `SEARCH_MODE`
and `ABSTAIN_THRESHOLD` to the best values you measured. Then continue:

```powershell
python src/build_index.py       # 5. build the final index with the chosen settings
python src/check_regression.py save   # 6. save today's recall@5 as the baseline
python src/run_answers.py 5     # 7. quick test of the full pipeline with the LLM (5 questions)
python src/run_answers.py       # 8. answer all questions, then grade the sample (see below)
python src/summarize_answers.py # 9. compute the answer-quality numbers
streamlit run app/streamlit_app.py    # 10. open the app in your browser
pytest                          # unit tests (no internet or model needed)
```

## The experiments

| Name | Chunk size / overlap | Search | Question it answers |
|---|---|---|---|
| E0 baseline | 800 / 150 | vector | How good is plain semantic search? |
| E1 | 500 / 100 and 1200 / 240 | vector | Does the chunk size matter? |
| E2 | all three sizes | hybrid | Does adding BM25 keywords help, especially for exact error names? |

A retrieved chunk counts as a **hit** when it comes from a gold page and contains at least one gold keyword
(see `data/eval/questions.jsonl`). This rule does not depend on chunk boundaries, so the chunk sizes can be compared fairly.
`experiments/details/<setting>.csv` lists every question with its hit rank - open it to see which questions fail and why.

Notes for reading the numbers:
- With about 40 answerable questions, one question changes recall by roughly 2.5 points. Small differences are noise;
  the results table includes a rough 95% range for recall@5.
- The embedding model only reads the first ~256 tokens of a chunk, so very large chunks lose their end for retrieval.
- The test questions were written by the project author. They are a sanity check, not a benchmark.

## Grading the answers (about 30 minutes)

`python src/run_answers.py` creates `experiments/answers.csv`. Open it in Excel and fill three columns (1 or 0)
**only for rows where `sample_for_grading` is True** (20 rows):

| Column | Give 1 when... |
|---|---|
| `manual_correct` | the answer covers the points in `expected_answer_points` and says nothing wrong |
| `manual_grounded` | every claim can be found in the passages that were retrieved (not from outside knowledge) |
| `manual_citations_ok` | the cited passages really support the claims they are attached to |

Save the file as CSV (keep the name), then run `python src/summarize_answers.py`.

## Results (fill in after your run)

| Setting | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| chunk800 vector (E0) | | | |
| chunk500 vector (E1) | | | |
| chunk1200 vector (E1) | | | |
| best hybrid (E2) | | | |

Abstention at threshold ___ : refuses ___ of unanswerable questions, wrongly refuses ___ of answerable ones.
Answer quality on ___ graded answers: correct ___, grounded ___, citations ok ___.

## Project structure

```
data/sources.csv            the 51 documentation pages (edit to add or remove pages)
data/eval/questions.jsonl   the test questions with gold pages and keywords
src/config.py               every setting in one place
src/fetch_docs.py           download pages          src/parse_docs.py   HTML -> text sections
src/chunk_docs.py           sections -> chunks      src/build_index.py  embeddings + saved index
src/search.py               vector + BM25 search    src/bm25.py         BM25 written from scratch
src/answer.py               LLM answer + checks     src/prompts.py      prompt text
src/run_eval.py             retrieval experiments   src/run_answers.py  answer all test questions
src/check_regression.py     recall@5 vs baseline    src/summarize_answers.py  answer-quality summary
app/streamlit_app.py        Ask / Evaluation / Corpus and usage
tests/                      unit tests              docs/SYSTEM_CARD.md  intended use and limits
```

## Keeping it up to date

The documentation changes with every release. To refresh: delete `data/raw`, run steps 1-2 and 5, then
`python src/check_regression.py`. A FAIL means recall dropped and the new pages or settings need a look.
Every index stores the embedding model, chunk settings and documentation versions in `indexes/final/index_info.json`.

## Limitations (please keep these in your write-up)

- Answers come only from 51 pages. Anything else (pandas, PyTorch, ...) should be refused, and how well that works is measured, not guaranteed.
- The citation check only verifies that a cited number points to a retrieved passage. Whether the passage supports the claim is checked by hand on a sample.
- Abstaining uses one similarity threshold. Questions that are close to the covered topics can still get through.
- Tracebacks are searched by their last line (error type and message) plus anything the user wrote before them.
- No real users: the usage log only shows what you tried yourself. This is a learning project, not a production service.
- Do not paste proprietary code into the app if you use a hosted LLM. By default the query log stores the first 80 characters of each question; set `LOG_QUESTION_TEXT = False` in `src/config.py` to store none.

## Optional: Docker

`Dockerfile` runs the app in a container (build after steps 1-5). It was written without being tested on a real machine.
