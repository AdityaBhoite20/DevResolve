# judge_answers.py - asks a second LLM to grade the answers (correct, grounded, citations ok)
#   python src/judge_answers.py        -> the 20 answers marked sample_for_grading in answers.csv
#   python src/judge_answers.py all    -> every answered question that has an answer in the documentation
# Set LLM_JUDGE_MODEL in .env to a DIFFERENT (ideally larger) model than the one that wrote the answers.
# These grades come from a model, not from a human. Always report them as "LLM-judged".
# If the daily limit is reached the script stops; run the same command again later and it continues.
import json  # to read and write files
import os  # to read the LLM_JUDGE_MODEL setting
import sys  # to read the command line
import time  # to pause between calls
import pandas as pd  # tables and CSV files
import config  # our settings (this also loads the .env file)
import search  # loading the index and searching
from answer import call_llm, llm_is_configured, DailyQuotaError  # the LLM call

JUDGE_PROMPT = """You are a strict grader of answers that were written from documentation passages.
You get a question, the points a good answer should cover, the answer, and the numbered passages the answer writer was allowed to use.
Grade three things. Each grade is 1 or 0.
correct: 1 if the answer covers the main expected points and says nothing that the passages contradict. Otherwise 0.
grounded: 1 only if every factual sentence in the answer is supported by the passages. A true statement that is not in the passages counts as 0.
citations_ok: 1 only if every citation like [2] points to a passage that supports the sentence it is attached to. If a citation points to an unrelated passage, or factual sentences have no citation, give 0.
Reply with JSON only, with no other text, for example: {"correct": 1, "grounded": 0, "citations_ok": 1, "reason": "one short sentence"}"""

if not llm_is_configured():  # we need the LLM for this script
    raise SystemExit("No LLM configured. Check your .env file.")

judge_model = os.getenv("LLM_JUDGE_MODEL", "")  # the model that grades
if judge_model == "" or judge_model.startswith("paste-"):  # no judge model was set
    judge_model = config.LLM_MODEL  # fall back to the answer model
    print("WARNING: LLM_JUDGE_MODEL is not set, so the model grades its own answers. Set it in .env for a fairer grade.")
print("Judge model:", judge_model)

judge_all = len(sys.argv) > 1 and sys.argv[1] == "all"  # grade everything instead of the 20-answer sample

answers = pd.read_csv(config.EXPERIMENTS_DIR / "answers.csv", encoding="utf-8-sig")  # the answers of the full run
answerable = answers["is_answerable"].astype(str).str.lower() == "true"  # questions with an answer in the corpus
answered = answers["abstained"].astype(str).str.lower() == "false"  # questions that got an answer
if judge_all:  # every answered question
    rows_to_judge = answers[answerable & answered]
else:  # only the sample
    marked = answers["sample_for_grading"].astype(str).str.lower() == "true"  # rows marked as the sample
    rows_to_judge = answers[marked & answerable & answered]
print("Answers to grade:", len(rows_to_judge))

questions = {}  # question id -> question dictionary
with open(config.QUESTIONS_FILE, encoding="utf-8") as f:  # open the question file
    for line in f:  # one JSON object per line
        if line.strip() != "":  # skip empty lines
            q = json.loads(line)  # convert text to a dictionary
            questions[q["question_id"]] = q  # remember it

cache_file = config.EXPERIMENTS_DIR / "judge_cache.jsonl"  # every finished grade is saved here at once
done = {}  # question id -> finished grade
if cache_file.exists():  # an earlier run saved some grades
    with open(cache_file, encoding="utf-8") as f:  # open the file
        for line in f:  # one JSON object per line
            if line.strip() != "":  # skip empty lines
                saved = json.loads(line)  # convert text to a dictionary
                if saved["judge_model"] == judge_model and saved["judge_correct"] is not None:  # only use readable grades from the same judge model
                    done[saved["question_id"]] = saved  # remember it (unreadable ones are graded again)

store = search.load_store(config.INDEX_DIR)  # load the final index (to show the judge the same passages the writer saw)

for number, (index, row) in enumerate(rows_to_judge.iterrows(), start=1):  # every answer to grade
    question_id = row["question_id"]  # for example q05
    if question_id in done:  # already graded
        continue
    q = questions[question_id]  # the question with its expected points
    output = search.search(q["question"], store, mode=config.SEARCH_MODE, top_k=config.TOP_K, domain="both")  # same retrieval as in the answer run
    lines = []  # the message for the judge
    lines.append("Question: " + q["question"])  # the question
    lines.append("Expected points: " + " ; ".join(q["expected_answer_points"]))  # what a good answer covers
    lines.append("")  # empty line
    lines.append("Answer to grade:")  # heading
    lines.append(str(row["answer"]))  # the answer
    lines.append("")  # empty line
    lines.append("Passages:")  # heading
    for passage_number, chunk in enumerate(output["results"], start=1):  # number the passages from 1
        lines.append("[" + str(passage_number) + "] " + chunk["title"] + " > " + chunk["section"])  # passage label
        lines.append(chunk["text"][:1500])  # passage text (same cut as in the answer prompt)
        lines.append("")  # empty line
    print("Grading", number, "of", len(rows_to_judge), ":", question_id)  # progress
    try:
        reply = call_llm("\n".join(lines), system_prompt=JUDGE_PROMPT, model=judge_model)  # ask the judge
    except DailyQuotaError as error:  # the daily limit is used up
        print("")
        print("STOPPED:", error)
        break  # leave the loop
    start = reply.find("{")  # where the JSON starts
    end = reply.rfind("}")  # where the JSON ends
    try:
        verdict = json.loads(reply[start:end + 1])  # read the JSON
        result = {
            "question_id": question_id,
            "judge_model": judge_model,
            "judge_correct": int(verdict["correct"]),
            "judge_grounded": int(verdict["grounded"]),
            "judge_citations_ok": int(verdict["citations_ok"]),
            "judge_reason": str(verdict.get("reason", ""))[:300],
        }  # the three grades and the reason
    except Exception:  # the judge did not reply with valid JSON
        result = {
            "question_id": question_id,
            "judge_model": judge_model,
            "judge_correct": None,
            "judge_grounded": None,
            "judge_citations_ok": None,
            "judge_reason": "could not read the judge reply: " + reply[:150].replace("\n", " "),
        }  # keep a note so we can look at it
    done[question_id] = result  # remember it
    with open(cache_file, "a", encoding="utf-8") as f:  # save it to disk at once
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    time.sleep(config.LLM_DELAY_SECONDS)  # pause so free plans do not block us

finished = []  # grades of the answers we wanted to grade
for index, row in rows_to_judge.iterrows():  # in the same order as answers.csv
    if row["question_id"] in done:  # it has a grade
        finished.append(done[row["question_id"]])
if len(finished) == 0:  # nothing was graded
    raise SystemExit("No grades were produced. Check the messages above.")

table = pd.DataFrame(finished)  # table of all grades
table.to_csv(config.EXPERIMENTS_DIR / "answers_judged.csv", index=False)  # save it
print("")
for index, row in table.iterrows():  # show every grade in one line
    print(row["question_id"], "| correct:", row["judge_correct"], "| grounded:", row["judge_grounded"], "| citations_ok:", row["judge_citations_ok"], "|", str(row["judge_reason"])[:90])
print("")
for column in ["judge_correct", "judge_grounded", "judge_citations_ok"]:  # the three averages
    values = pd.to_numeric(table[column], errors="coerce").dropna()  # only the readable grades
    if len(values) > 0:
        print(column, "=", round(float(values.mean()), 3), "(n =", len(values), ")")
print("Saved experiments/answers_judged.csv. Now run: python src/summarize_answers.py")
if len(finished) < len(rows_to_judge):  # we stopped early
    print("NOT COMPLETE:", len(finished), "of", len(rows_to_judge), "graded. Run the same command again later to continue.")
