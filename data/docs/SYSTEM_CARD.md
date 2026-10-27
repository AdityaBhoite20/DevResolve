# DevResolve system card

Short description of what the system is for, what it was tested on and where it fails.
Items marked (fill in) should be completed with your own numbers after running the project.

## Intended use
Help students and junior developers understand Python errors and common scikit-learn problems, with answers that
point to the official documentation. Learning and portfolio project.

## Not intended for
- Production debugging of proprietary code, security-sensitive or legal/medical decisions.
- Any library other than Python's standard documentation pages and the scikit-learn pages listed in `data/sources.csv`.

## Data
- 51 pages: official Python documentation and scikit-learn documentation (list in `data/sources.csv`).
- Pages are downloaded for personal use, split into sections and chunks; every answer links back to the source page.
- Versions and download date are stored per chunk and in `indexes/final/index_info.json`. (fill in: scikit-learn ___, Python ___, downloaded ___)

## Method
Sentence-Transformers embeddings (`all-MiniLM-L6-v2`, run by PyTorch), exact vector search (FAISS if installed, otherwise numpy),
BM25 keyword search written from scratch, rank merging with reciprocal rank fusion, an LLM that must cite numbered passages,
a similarity threshold for abstaining, and a citation-number check.

## Evaluation
- Test set: fixed questions in `data/eval/questions.jsonl` (answerable and unanswerable), written by the project author.
- Retrieval: recall@k and MRR, chunk size and search method compared (see `experiments/results.csv`). (fill in best setting and numbers)
- Abstention: refuse rate on unanswerable questions vs wrongly refused answerable questions at the chosen threshold. (fill in)
- Answers: manual grading of a 20-answer sample for correctness, groundedness and citation support. (fill in)
- Small test set: one question moves recall by about 2.5 points, so small differences are not conclusive.

## Known failure modes
(fill in with 3 real examples from `experiments/details/*.csv` and `experiments/answers.csv`, for example: a question whose
right page was found but the wrong section, a question that should have been refused but was answered, an answer with a weak citation.)

## Risks and mitigations
- Wrong or invented answers: the prompt requires citations and allows "insufficient evidence"; a sample is graded by hand.
- Out-of-scope questions: similarity threshold plus the LLM's own insufficient-evidence reply.
- Privacy: do not paste proprietary code when a hosted LLM is used; the query log keeps at most 80 characters of a question.
- Outdated documentation: re-download, rebuild and run `python src/check_regression.py` after each documentation release.

## Monitoring
`logs/queries.csv` records time, top similarity, abstained or not, retrieval and LLM time and citation validity.
The app's "Corpus and usage" tab shows these. Regression check: `python src/check_regression.py`.
