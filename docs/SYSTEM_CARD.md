# DevResolve system card

What the system is for, what it was tested on, and where it fails.

## Intended use
Help students and junior developers understand Python errors and common scikit-learn problems, with answers that
point to the official documentation. Learning and portfolio project.

## Not intended for
- Production debugging of proprietary code, security-sensitive or legal/medical decisions.
- Any library other than Python's standard documentation pages and the scikit-learn pages listed in `data/sources.csv`.

## Data
- 51 pages of official documentation (scikit-learn 1.9.1, Python 3.15.2), downloaded October 2026 for personal use, split into sections
  and chunks (2,583 chunks at chunk size 800). Every answer links back to the source page.
- Versions and download date are stored per chunk and in `indexes/final/index_info.json`.

## Method
Sentence-Transformers embeddings (`all-MiniLM-L6-v2`, run by PyTorch), exact vector search (FAISS), BM25 keyword search written from scratch,
rank merging with reciprocal rank fusion, an LLM (`openai/gpt-oss-20b` on Groq) that must cite numbered passages,
a similarity threshold for abstaining (0.45), a one-time retry when an answer has no citations (never needed in the final run),
and a citation-number check.

## Evaluation (details and tables in README, section Results)
- Test set: 51 questions written by the project author (42 answerable, 9 unanswerable) with gold pages and keywords. Small and not independent:
  one question moves recall by about 2.4 points, and the threshold and several fixes were chosen using these same questions.
- Retrieval: recall@5 = 0.952 (40/42) for chunk size 800 with hybrid search; vector only gives 0.905.
- Abstention: 8 of 9 unanswerable questions refused, 2 of 42 answerable questions wrongly refused.
- Answers: 40/40 have valid citation numbers. On a 20-answer sample graded by a second LLM (`openai/gpt-oss-120b`, not by a human):
  correct 17/20, every claim supported by the passages 14/20, citations support their sentences 13/20.

## Known failure modes (all seen in the test run)
1. **Near-miss topics are answered.** "How do I tune XGBoost hyperparameters?" scores 0.51, passes the threshold, and the model answers from scikit-learn's
   gradient boosting passages ("XGBoost-style"), with citations that do not support a claim about XGBoost. Other out-of-scope questions (pandas, PyTorch, Django,
   TensorFlow, asyncio, general knowledge) were refused.
2. **Valid questions refused.** A short pasted traceback (`KeyError: 'apple'`, similarity 0.35) and "Why do I get different results every time I run my model?"
   (similarity 0.3986) fall below the threshold and are refused although the documentation covers them. Short queries score lower.
3. **Right page, wrong section.** For "How does Python decide where to look for a module when I use import?" the import-system page was retrieved,
   but not the section about `sys.path`, so the answer missed that point.
4. **Claims from outside the passages.** In the judged sample 6 of 20 answers contained statements that are true but not in the retrieved passages
   (for example generic advice), and 7 of 20 had a citation that did not support its sentence.
5. **Fixed during the project:** a wrong page address (one page was never indexed), a bug that searched with the user's question instead of the error line
   when a question followed a traceback, and answers without citations.

## Risks and mitigations
- Wrong or invented answers: the prompt requires citations and allows "insufficient evidence"; a second model grades a sample; limits are listed above.
- Out-of-scope questions: similarity threshold plus the model's own insufficient-evidence reply (neither is perfect, see failure mode 1).
- Privacy: do not paste proprietary code when a hosted LLM is used; the query log keeps at most 80 characters of a question.
- Outdated documentation: re-download, rebuild and run `python src/check_regression.py` after each documentation release.

## Monitoring
`logs/queries.csv` records time, top similarity, abstained or not, retrieval and LLM time and citation validity.
The app's "Corpus and usage" tab shows these. Regression check: `python src/check_regression.py`.
