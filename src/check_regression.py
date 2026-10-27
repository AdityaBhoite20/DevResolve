# check_regression.py - re-runs the retrieval test on the FINAL index and compares recall@5 with a saved baseline
#   python src/check_regression.py save    -> saves the current recall@5 as the baseline
#   python src/check_regression.py         -> compares the current recall@5 with the baseline
import json  # to read and write files
import sys  # to read the command line and set the exit code
from datetime import datetime  # time stamp
import config  # our settings
import search  # loading the index and searching
from metrics import first_hit_rank, recall_at_k  # metrics

TOLERANCE = 0.03  # a drop of up to 3 percentage points is accepted (one question is worth about 2.5 points)

store = search.load_store(config.INDEX_DIR)  # load the final index
questions = []  # the test questions
with open(config.QUESTIONS_FILE, encoding="utf-8") as f:  # open the question file
    for line in f:  # one JSON object per line
        if line.strip() != "":  # skip empty lines
            questions.append(json.loads(line))  # convert text to a dictionary

ranks = []  # hit rank of every answerable question
for q in questions:  # every question
    if not q["is_answerable"]:  # skip questions that have no answer in the corpus
        continue
    output = search.search(q["question"], store, mode=config.SEARCH_MODE, top_k=10, domain="both")  # retrieve
    ranks.append(first_hit_rank(output["results"], q))  # remember the rank of the first hit

current = recall_at_k(ranks, 5)  # recall@5 now
print("Current recall@5:", round(current, 3), "on", len(ranks), "answerable questions (mode:", config.SEARCH_MODE + ")")

baseline_file = config.EXPERIMENTS_DIR / "baseline.json"  # where the baseline is stored
config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)  # create the folder if needed

if len(sys.argv) > 1 and sys.argv[1] == "save":  # the user wants to save a new baseline
    baseline = {
        "recall_at_5": current,
        "mode": config.SEARCH_MODE,
        "chunk_size": config.CHUNK_SIZE,
        "n_questions": len(ranks),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }  # what we store
    with open(baseline_file, "w", encoding="utf-8") as f:  # write the file
        json.dump(baseline, f, indent=2)
    print("Baseline saved to", baseline_file)
else:  # compare with the baseline
    if not baseline_file.exists():  # nothing to compare with
        raise SystemExit("No baseline yet. Run: python src/check_regression.py save")
    with open(baseline_file, encoding="utf-8") as f:  # read the baseline
        baseline = json.load(f)
    print("Baseline recall@5:", round(baseline["recall_at_5"], 3), "(saved", baseline["saved_at"] + ")")
    if current < baseline["recall_at_5"] - TOLERANCE:  # quality dropped too much
        print("FAIL: recall@5 dropped by more than", TOLERANCE)
        sys.exit(1)  # a non-zero exit code means failure
    print("PASS: recall@5 is within", TOLERANCE, "of the baseline")
