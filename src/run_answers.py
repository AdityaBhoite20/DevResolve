# run_answers.py - answers the test questions with the full pipeline and saves experiments/answers.csv
#   python src/run_answers.py          -> all questions (continues where an earlier run stopped)
#   python src/run_answers.py 5        -> only the first 5 questions (good for a quick test)
#   python src/run_answers.py fresh    -> forget earlier answers and start again (do this after changing settings or rebuilding the index)
#   python src/run_answers.py redo q03 q46   -> answer only these questions again (good for testing a prompt change)
# If the LLM daily limit is reached, the script stops cleanly. Run the same command again later and it continues.
# Grade the rows in answers.csv only AFTER the run is complete, then run summarize_answers.py
import json  # to read the questions and the progress file
import shutil  # to copy files
import sys  # to read the command line
import time  # to pause between LLM calls
import pandas as pd  # tables and CSV files
import config  # our settings
import search  # loading the index
from answer import answer_question, llm_is_configured, DailyQuotaError  # the pipeline
from metrics import first_hit_rank  # to check if the right evidence was retrieved

if not llm_is_configured():  # we need the LLM for this script
    raise SystemExit("No LLM configured. Copy .env.example to .env and fill in LLM_API_KEY and LLM_MODEL.")

limit = None  # by default we answer every question
fresh = False  # by default we continue an earlier run
redo_ids = []  # questions that must be answered again
for argument in sys.argv[1:]:  # read the words typed after the script name
    if argument == "fresh":  # start again
        fresh = True
    elif argument.isdigit():  # a number limits the questions
        limit = int(argument)
    elif argument.startswith("q") and argument[1:].isdigit():  # a question id such as q46
        redo_ids.append(argument)

config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)  # create the folder if needed
cache_file = config.EXPERIMENTS_DIR / "answers_cache.jsonl"  # every finished question is saved here at once
if fresh and cache_file.exists():  # the user wants to start again
    cache_file.unlink()  # delete the progress file

done = {}  # question id -> finished row
if cache_file.exists():  # an earlier run saved some answers
    with open(cache_file, encoding="utf-8") as f:  # open the progress file
        for line in f:  # one JSON object per line
            if line.strip() != "":  # skip empty lines
                row = json.loads(line)  # convert text to a dictionary
                done[row["question_id"]] = row  # remember it
for question_id in redo_ids:  # forget the old answer of these questions
    done.pop(question_id, None)
print("Questions already answered in earlier runs:", len(done))

store = search.load_store(config.INDEX_DIR)  # load the final index
questions = []  # the test questions
with open(config.QUESTIONS_FILE, encoding="utf-8") as f:  # open the question file
    for line in f:  # one JSON object per line
        if line.strip() != "":  # skip empty lines
            questions.append(json.loads(line))  # convert text to a dictionary
if limit is not None:  # the user gave a limit
    questions = questions[:limit]  # keep only the first N questions

stopped_early = False  # becomes True if the daily limit stops us
for number, q in enumerate(questions, start=1):  # every question
    if q["question_id"] in done:  # we already have this answer
        continue  # skip it
    print("Question", number, "of", len(questions), ":", q["question_id"])  # progress
    try:
        output = answer_question(q["question"], store, config.SEARCH_MODE, config.TOP_K, "both")  # run the pipeline
    except DailyQuotaError as error:  # the daily limit is used up
        print("")
        print("STOPPED:", error)
        stopped_early = True  # remember that we stopped
        break  # leave the loop
    cited_urls = []  # pages that the answer cited
    for cited_id in output["cited_ids"]:  # every valid citation number
        cited_urls.append(output["results"][cited_id - 1]["source_url"].replace("https://", ""))  # its page
    if q["is_answerable"]:  # we can check the retrieval
        gold_in_context = first_hit_rank(output["results"], q) is not None  # was a correct chunk among the passages?
    else:
        gold_in_context = ""  # not applicable
    row = {
        "question_id": q["question_id"],
        "question_type": q["question_type"],
        "is_answerable": q["is_answerable"],
        "question": q["question"],
        "expected_answer_points": " ; ".join(q["expected_answer_points"]),
        "answer": output["answer"],
        "abstained": output["abstained"],
        "abstain_reason": output["abstain_reason"],
        "top_vector_score": round(output["top_vector_score"], 4),
        "gold_in_context": gold_in_context,
        "citation_ok": output["citation_ok"],
        "cited_ids": ",".join([str(i) for i in output["cited_ids"]]),
        "invalid_ids": ",".join([str(i) for i in output["invalid_ids"]]),
        "cited_urls": " | ".join(cited_urls),
        "retrieval_ms": round(output["retrieval_ms"], 1),
        "llm_seconds": round(output["llm_seconds"], 2),
        "retries": output["retries"],
    }  # everything about this question
    done[q["question_id"]] = row  # remember it
    with open(cache_file, "a", encoding="utf-8") as f:  # save it to disk immediately
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    time.sleep(config.LLM_DELAY_SECONDS)  # pause so free plans do not block us

rows = []  # finished rows in question order
for q in questions:  # every question
    if q["question_id"] in done:  # it has an answer
        rows.append(done[q["question_id"]])
if len(rows) == 0:  # nothing finished at all
    raise SystemExit("No answers were produced. Check the messages above.")

df = pd.DataFrame(rows)  # table of all finished answers
df["sample_for_grading"] = False  # by default a row is not part of the grading sample
candidates = df[(df["is_answerable"] == True) & (df["abstained"] == False)]  # answers that are worth grading
n_sample = min(20, len(candidates))  # grade at most 20 answers
if n_sample > 0:  # there is something to grade
    sample = candidates.sample(n=n_sample, random_state=42)  # a random but repeatable sample
    df.loc[sample.index, "sample_for_grading"] = True  # mark the sampled rows
df["manual_correct"] = ""  # YOU fill this: 1 = answer covers the expected points, 0 = it does not
df["manual_grounded"] = ""  # YOU fill this: 1 = every claim is supported by the retrieved passages, 0 = not
df["manual_citations_ok"] = ""  # YOU fill this: 1 = cited passages support the claims, 0 = not
df["notes"] = ""  # optional comments

answers_file = config.EXPERIMENTS_DIR / "answers.csv"  # the file you grade
if answers_file.exists():  # an older file exists
    shutil.copy(answers_file, config.EXPERIMENTS_DIR / "answers_previous.csv")  # keep a backup so grades are never lost
df.to_csv(answers_file, index=False, encoding="utf-8-sig")  # save (utf-8-sig opens nicely in Excel)
print("")
print("Saved experiments/answers.csv with", len(rows), "of", len(questions), "questions")
if stopped_early or len(rows) < len(questions):
    print("NOT COMPLETE. Run the same command again later to continue. Do not grade yet.")
else:
    print("Complete. Now grade the", n_sample, "rows where sample_for_grading is True (columns manual_correct, manual_grounded, manual_citations_ok), then run: python src/summarize_answers.py")
