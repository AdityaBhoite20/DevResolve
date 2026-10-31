# DevResolve

A small RAG (retrieval-augmented generation) system for **Python errors** and **scikit-learn questions**.
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
python src/judge_answers.py     # 9. (optional) let a second LLM grade 20 answers, then run summarize_answers.py
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

## Grading the answers

**Option A: LLM judge (no manual work, about 10 minutes).** Put a second, larger model in `.env` as `LLM_JUDGE_MODEL`, then run
`python src/judge_answers.py` and `python src/summarize_answers.py`. A model grades 20 sampled answers for correctness,
groundedness and citation support, looking at the same passages the answer writer saw. These are **LLM-judged grades, not human grades**:
say so in any write-up, and expect the judge to make mistakes.

**Option B: manual grading (about 30 minutes, more trustworthy).**

`python src/run_answers.py` creates `experiments/answers.csv`. Open it in Excel and fill three columns (1 or 0)
**only for rows where `sample_for_grading` is True** (20 rows):

| Column | Give 1 when... |
|---|---|
| `manual_correct` | the answer covers the points in `expected_answer_points` and says nothing wrong |
| `manual_grounded` | every claim can be found in the passages that were retrieved (not from outside knowledge) |
| `manual_citations_ok` | the cited passages really support the claims they are attached to |

Save the file as CSV (keep the name), then run `python src/summarize_answers.py`.

## Results

Measured on 51 hand-written questions (42 answerable, 9 that should be refused) over 51 documentation pages
(scikit-learn 1.9.1 and Python 3.15.2 documentation, downloaded October 2026). The answer model was `openai/gpt-oss-20b`
on Groq; answers were graded by `openai/gpt-oss-120b`.

**Retrieval** (a hit = a chunk from the right page that contains a gold keyword; recall@5 range = rough 95% bootstrap range):

| Setting | Recall@1 | Recall@5 (range) | MRR@10 |
|---|---|---|---|
| chunk 500, vector | 0.643 | 0.786 (0.667-0.905) | 0.719 |
| chunk 500, hybrid | 0.643 | 0.881 (0.786-0.976) | 0.740 |
| chunk 800, vector (E0 baseline setting) | 0.738 | 0.905 (0.810-0.976) | 0.810 |
| **chunk 800, hybrid (chosen)** | 0.714 | **0.952 (0.881-1.000)** | 0.801 |
| chunk 1200, vector | 0.738 | 0.905 (0.810-0.976) | 0.816 |
| chunk 1200, hybrid | 0.690 | 0.929 (0.833-1.000) | 0.803 |

Hybrid search finds the right passage in the top 5 for 2 more questions than vector search at every chunk size, but it ranks the
right passage first slightly less often. The answer model sees the top 5 passages, so recall@5 is the metric that matters here.

**How recall@5 improved at chunk size 800** (the first two steps were changes of method, the last one was bug fixing):

| Step | Vector | Hybrid |
|---|---|---|
| First run, plain chunks | 0.833 (35/42) | 0.881 (37/42) |
| Every exception or class definition gets its own chunk | 0.881 (37/42) | 0.929 (39/42) |
| Fixed a wrong page address and a traceback parsing bug | 0.905 (38/42) | 0.952 (40/42) |

**Abstaining** (similarity threshold 0.45, tuned on the same questions, so optimistic): 8 of 9 unanswerable questions are refused,
2 of 42 answerable questions are wrongly refused.

**Answers** (full run, 40 answers written): every answer has citations that point to a retrieved passage (first version of the prompt: 54%).
The right evidence was among the retrieved passages for 38 of 40 answers.
**LLM-judged sample of 20 answers (not human grades):** correct 0.85 (rough 95% range 0.64-0.95), every claim supported by the passages 0.70 (0.48-0.85),
citations support their sentences 0.65 (0.43-0.82). A valid citation number is not the same as a supported claim: the citation checker only proves
the number points to a retrieved passage.

## Project structure

```
data/sources.csv            the 51 documentation pages (edit to add or remove pages)
data/eval/questions.jsonl   the test questions with gold pages and keywords
src/config.py               every setting in one place
src/fetch_docs.py           download pages
src/parse_docs.py           HTML -> text sections       
src/chunk_docs.py           sections -> chunks
src/build_index.py          embeddings + saved index      
src/search.py               vector + BM25 search
src/bm25.py                 BM25 written from scratch    
src/answer.py               LLM answer + checks
src/prompts.py              prompt text     
src/run_eval.py             retrieval experiments
src/run_answers.py          answer all test questions
src/summarize_answers.py    answer-quality summary   
src/check_regression.py     recall@5 vs baseline    
app/streamlit_app.py        Ask / Evaluation / Corpus and usage
tests/                      unit tests      
```

## Keeping it up to date

The documentation changes with every release. To refresh: delete `data/raw`, run steps 1-2 and 5, then
`python src/check_regression.py`. A FAIL means recall dropped and the new pages or settings need a look.
Every index stores the embedding model, chunk settings and documentation versions in `indexes/final/index_info.json`.

## Limitations 

- Answers come only from 51 pages. Anything else (pandas, PyTorch, ...) should be refused, and how well that works is measured, not guaranteed.
- The citation check only verifies that a cited number points to a retrieved passage. Whether the passage supports the claim is checked by hand on a sample.
- Abstaining uses one similarity threshold. Questions that are close to the covered topics can still get through.
- Tracebacks are searched by their last line (error type and message) plus anything the user wrote before them.
- No real users: the usage log only shows what you tried yourself. This is a learning project, not a production service.
- Do not paste proprietary code into the app if you use a hosted LLM. By default the query log stores the first 80 characters of each question; set `LOG_QUESTION_TEXT = False` in `src/config.py` to store none.

