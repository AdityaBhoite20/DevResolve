# config.py - all settings for DevResolve live in this one file
import os  # to read environment variables (API keys)
from pathlib import Path  # to build file paths
from dotenv import load_dotenv  # to read the .env file

# ---------- folders ----------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root folder (one level above src/)
DATA_DIR = BASE_DIR / "data"  # all data lives here
RAW_DIR = DATA_DIR / "raw"  # downloaded HTML pages
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned documents and chunks
EVAL_DIR = DATA_DIR / "eval"  # the test questions
SOURCES_FILE = DATA_DIR / "sources.csv"  # list of pages to download
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"  # the fixed test set
INDEX_DIR = BASE_DIR / "indexes" / "final"  # the index used by the app
EXPERIMENT_INDEX_DIR = BASE_DIR / "indexes" / "experiments"  # cached embeddings for experiments
EXPERIMENTS_DIR = BASE_DIR / "experiments"  # result CSV files
LOGS_DIR = BASE_DIR / "logs"  # query log used by the monitoring section

load_dotenv(BASE_DIR / ".env")  # load secrets from the .env file (this file is never committed)

# ---------- retrieval settings ----------
# NOTE: the values below are a starting point. After you run src/run_eval.py,
# change them to the best settings you measured (then run src/build_index.py again).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small embedding model that runs on a laptop (PyTorch)
CHUNK_SIZE = 800  # maximum characters per chunk (a soft limit)
CHUNK_OVERLAP = 150  # characters repeated between neighbouring chunks
MIN_CHUNK_CHARS = 40  # chunks shorter than this are thrown away
SEARCH_MODE = "hybrid"  # "vector" = meaning only, "hybrid" = meaning + BM25 keywords
TOP_K = 5  # how many chunks we give to the LLM
ABSTAIN_THRESHOLD = 0.45  # if the best similarity score is below this, we say "insufficient evidence"

# ---------- logging ----------
LOG_QUESTION_TEXT = True  # False = the query log stores only the question length, not its text

# ---------- LLM settings (read from the .env file) ----------
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # your provider API key
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # empty = OpenAI, or the URL of another provider
LLM_MODEL = os.getenv("LLM_MODEL", "")  # the model name from your provider
LLM_DELAY_SECONDS = float(os.getenv("LLM_DELAY_SECONDS", "2"))  # pause between calls in batch runs
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "")  # optional, for example "low"; leave empty if you are not sure
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))  # how long one LLM call may take (use a bigger number, like 900, for a local model)