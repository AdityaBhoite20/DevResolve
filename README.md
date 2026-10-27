# DevResolve | RAG-Based Developer Troubleshooting System

DevResolve is a learning-oriented project exploring how **Retrieval-Augmented Generation (RAG)** can help answer Python troubleshooting and scikit-learn questions using a curated technical documentation corpus.

Instead of relying entirely on an LLM's existing knowledge, the system retrieves relevant documentation passages and provides them as context for generating answers with numbered citations. It also includes an insufficient-evidence mechanism for questions that fall outside the available documentation.

The main focus is understanding and evaluating the RAG pipeline, particularly document chunking, semantic retrieval, keyword search, answer grounding, and handling unsupported questions.

## How it works

```text
Technical Documentation
          |
          v
   Text Extraction
          |
          v
    Text Chunking
          |
          v
   Embedding Generation
          |
          v
      User Query
          |
          v
   Retrieve Relevant Passages
   (Vector Search + BM25)
          |
          v
    Merge Rankings (RRF)
          |
          v
   LLM Answer Generation
          |
          v
  Answer with Source Citations
          |
          v
  Citation and Evidence Checks
```

If the retrieved evidence is insufficient, the system can abstain rather than provide an unsupported answer. This behavior is evaluated, not assumed to be perfect.

## Tech stack

- **Language:** Python
- **Embeddings:** Sentence Transformers, PyTorch
- **Retrieval:** Semantic vector search, custom BM25 keyword search
- **Ranking:** Reciprocal Rank Fusion (RRF)
- **LLM integration:** OpenAI-compatible API
- **Interface:** Streamlit
- **Evaluation and testing:** pandas, pytest, custom evaluation scripts

FAISS is optional; the system can fall back to NumPy-based vector search.

## Key features

- **Documentation ingestion:** Downloads and parses selected official Python and scikit-learn documentation pages.
- **Configurable chunking:** Compares different chunk sizes and overlap settings to understand their effect on retrieval.
- **Hybrid retrieval:** Combines semantic search with BM25 keyword search to improve the retrieval of relevant passages, including technical terms and error names.
- **Grounded generation:** Uses retrieved passages as context and prompts the LLM to provide source-cited answers.
- **Insufficient-evidence handling:** Uses a configurable similarity threshold and an explicit abstention response for unsupported questions.
- **Evaluation pipeline:** Compares retrieval configurations using a fixed question set and provides scripts for reviewing answer correctness, grounding, and citations.

## Evaluation and results

The project uses a **51-question evaluation set**, consisting of 42 questions answerable from the indexed documentation and 9 questions intended to be refused.

Experiments compare chunk sizes of 500, 800, and 1,200, with vector search and hybrid retrieval.

### Latest retrieval results

| Configuration | Relevant questions retrieved in the top 5 |
|---|---:|
| Vector search, chunk size 800 | 38/42 |
| Hybrid retrieval, chunk size 800 | 40/42 |

In the latest corrected run, hybrid retrieval found relevant passages for **2 additional questions** compared with vector search at the same chunk size.

The results also showed that chunking and data-quality fixes matter: correcting a missing documentation URL and improving traceback parsing helped resolve earlier retrieval failures.

### Abstention experiment

The abstention threshold was tuned to **0.45** on the current test set. At this setting, the system correctly refused 8 of the 9 intended-unanswerable questions while incorrectly refusing 2 answerable questions.

These results are preliminary and measured on the same evaluation set used for configuration selection. They should not be interpreted as independent benchmark performance or a guarantee of correct answers.

### Evaluation limitations

- The test questions and gold keywords were created for this project; they are not an external benchmark.
- A retrieval hit is determined using the expected source page and a matching keyword. This is a practical evaluation rule, not a complete measure of semantic relevance.
- Citation presence and citation correctness are different. The system checks citation references, while whether a passage actually supports a claim requires manual review.
- The LLM can still answer questions about related but unsupported topics. Abstention reduces this risk but does not eliminate it.

## Setup

### Prerequisites

- Python 3.10–3.12
- Git
- An internet connection to download documentation and the embedding model
- An API key for an OpenAI-compatible LLM provider if you want generated answers

### Installation

Run the following commands in PowerShell from the project directory:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install faiss-cpu
copy .env.example .env
```

FAISS is optional. If its installation fails, the NumPy fallback can be used.

Configure `.env` with the LLM API key, model name, and base URL required by your chosen provider. Refer to `.env.example` for the expected variable names.

### Run the pipeline

Run the following steps in order:

```powershell
# 1. Download the documentation
python src/fetch_docs.py

# 2. Parse the downloaded pages
python src/parse_docs.py

# 3. Inspect chunk statistics (optional)
python src/chunk_docs.py

# 4. Run retrieval experiments
python src/run_eval.py

# 5. Build the final index
python src/build_index.py

# 6. Save a retrieval regression baseline
python src/check_regression.py save

# 7. Test a small sample of questions
python src/run_answers.py 5

# 8. Generate answers for the evaluation set
python src/run_answers.py

# 9. Summarize reviewed answer-quality results
python src/summarize_answers.py

# 10. Launch the Streamlit interface
streamlit run app/streamlit_app.py
```

Run the unit tests with:

```powershell
pytest
```

The retrieval pipeline and tests can be run without an LLM API key, but generating answers requires an available LLM provider.

## Manual answer evaluation

The answer-generation script creates `experiments/answers.csv`. For rows marked `sample_for_grading`, manually review the sampled answers and fill in the corresponding columns with `1` or `0`.

| Field | What it evaluates |
|---|---|
| `manual_correct` | Whether the answer is correct and covers the expected answer points |
| `manual_grounded` | Whether the answer's claims are supported by the retrieved passages |
| `manual_citations_ok` | Whether the cited passages actually support the associated claims |

Save the reviewed CSV and run:

```powershell
python src/summarize_answers.py
```

Manual review is important because the presence of a citation does not establish that the cited passage supports the answer.

## Project structure

```text
DevResolve/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── docs/
│   │   └── SYSTEM_CARD.md
│   ├── eval/
│   │   └── questions.jsonl
│   └── sources.csv
├── experiments/
│   ├── details/
│   ├── answer_summary.csv
│   ├── answers.csv
│   ├── baseline.json
│   ├── results.csv
│   ├── results_by_type.csv
│   └── threshold_sweep.csv
├── src/
│   ├── answer.py
│   ├── bm25.py
│   ├── build_index.py
│   ├── check_regression.py
│   ├── chunk_docs.py
│   ├── citations.py
│   ├── config.py
│   ├── fetch_docs.py
│   ├── metrics.py
│   ├── parse_docs.py
│   ├── prompts.py
│   ├── run_answers.py
│   ├── run_eval.py
│   ├── search.py
│   └── summarize_answers.py
├── tests/
├── .env.example
├── .github/
│   └── workflows/
│       └── tests.yml
├── requirements.txt
└── README.md
```

## Limitations and future improvements

- The indexed corpus is limited to selected Python and scikit-learn documentation pages; questions outside this scope may not be answerable.
- Retrieval quality depends on parsing, chunk boundaries, embedding similarity, and ranking.
- The similarity threshold is a heuristic, not a calibrated probability that an answer is correct.
- The current evaluation set is relatively small and authored for this project.
- Answer correctness and citation support require further manual evaluation.
- The system is a learning and experimentation project, not a production troubleshooting service.

Potential next steps include expanding the evaluation set, improving retrieval for near-miss topics, and testing answer quality across additional LLMs.

## Project objective

The goal of DevResolve is to understand the internal workflow of a RAG system and investigate how retrieval choices affect the quality and reliability of LLM-generated answers.

The emphasis is on **measurable retrieval behavior, evidence-grounded generation, and transparent limitations**, rather than claiming that the system can reliably solve every developer question.