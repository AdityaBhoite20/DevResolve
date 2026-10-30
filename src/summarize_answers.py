# summarize_answers.py - turns experiments/answers.csv (after your manual grading) into summary numbers
# run it with:  python src/summarize_answers.py
import pandas as pd  # tables and CSV files
import config  # our settings

df = pd.read_csv(config.EXPERIMENTS_DIR / "answers.csv", encoding="utf-8-sig")  # read the answers file
df["abstained"] = df["abstained"].astype(str).str.lower() == "true"  # make sure it is a True/False column (Excel writes TRUE)
df["is_answerable"] = df["is_answerable"].astype(str).str.lower() == "true"  # same here

rows = []  # summary rows: (metric, value, n)

answerable = df[df["is_answerable"]]  # questions that have an answer in the corpus
unanswerable = df[~df["is_answerable"]]  # questions that should be refused
if len(unanswerable) > 0:  # there are unanswerable questions
    rows.append(["abstained_correctly_on_unanswerable", round(float(unanswerable["abstained"].mean()), 3), len(unanswerable)])
if len(answerable) > 0:  # there are answerable questions
    rows.append(["abstained_wrongly_on_answerable", round(float(answerable["abstained"].mean()), 3), len(answerable)])

answered = answerable[~answerable["abstained"]]  # answerable questions that got an answer
if len(answered) > 0:  # there are answers
    citation_flags = answered["citation_ok"].astype(str).str.lower() == "true"  # automatic check: all citations point to real passages
    rows.append(["answers_with_valid_citations_auto", round(float(citation_flags.mean()), 3), len(answered)])
    if "retries" in answered.columns:  # the column exists in newer runs
        retries = pd.to_numeric(answered["retries"], errors="coerce").fillna(0)  # missing values count as 0
        rows.append(["answers_that_needed_a_citation_retry", round(float((retries > 0).mean()), 3), len(answered)])
    context_flags = answered["gold_in_context"].astype(str).str.lower() == "true"  # was the right evidence in the passages?
    rows.append(["answers_with_gold_evidence_in_context", round(float(context_flags.mean()), 3), len(answered)])

graded = df[df["sample_for_grading"].astype(str).str.lower() == "true"]  # the rows you were asked to grade
for column in ["manual_correct", "manual_grounded", "manual_citations_ok"]:  # the three manual metrics
    values = pd.to_numeric(graded[column], errors="coerce").dropna()  # keep only the numbers you typed
    if len(values) > 0:  # at least one row graded
        rows.append([column, round(float(values.mean()), 3), len(values)])
    else:
        print("Note:", column, "has no manual grades (fine if you use judge_answers.py instead)")  # reminder

judged_file = config.EXPERIMENTS_DIR / "answers_judged.csv"  # written by judge_answers.py
if judged_file.exists():  # an LLM judge graded some answers
    judged = pd.read_csv(judged_file)  # read the grades
    for column in ["judge_correct", "judge_grounded", "judge_citations_ok"]:  # the three LLM-judged metrics
        values = pd.to_numeric(judged[column], errors="coerce").dropna()  # only the readable grades
        if len(values) > 0:
            rows.append(["llm_" + column, round(float(values.mean()), 3), len(values)])  # these are LLM-judged, not human grades

summary = pd.DataFrame(rows, columns=["metric", "value", "n"])  # build the summary table
summary.to_csv(config.EXPERIMENTS_DIR / "answer_summary.csv", index=False)  # save it
print(summary.to_string(index=False))  # print it so you can copy it
print("")
print("Saved experiments/answer_summary.csv")
