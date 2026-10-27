# answer.py - finds passages, asks the LLM for an answer, checks the citations and logs the query
import csv  # to write the query log
import time  # to measure the LLM time
from datetime import datetime  # for the log timestamp
import config  # our settings
from search import search  # the retrieval function
from prompts import SYSTEM_PROMPT, RETRY_NOTE, build_user_prompt  # the prompt texts
from citations import find_cited_ids  # citation checker


class DailyQuotaError(Exception):
    pass  # raised when the provider says the daily request limit is used up


def llm_is_configured():
    # True when the .env file contains a real API key and a model name (the placeholder text from .env.example does not count)
    if config.LLM_API_KEY == "" or config.LLM_MODEL == "":  # something is missing
        return False
    if config.LLM_API_KEY.startswith("paste-") or config.LLM_MODEL.startswith("paste-"):  # the example text was not replaced
        return False
    return True


def call_llm(user_prompt):
    # sends the prompt to the LLM and returns the answer text
    from openai import OpenAI  # imported here so the rest of the project works without the library
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL or None, max_retries=0, timeout=config.LLM_TIMEOUT_SECONDS)  # we do the retrying ourselves, so every problem is visible
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]  # the conversation we send
    settings = {"temperature": 0}  # extra settings for the call (temperature 0 makes answers more repeatable)
    if config.LLM_REASONING_EFFORT != "":  # optional: fewer "thinking" tokens = faster answers
        settings["reasoning_effort"] = config.LLM_REASONING_EFFORT
    last_error = None  # remembers the last problem
    for attempt in range(4):  # try up to 4 times (free plans sometimes limit the speed)
        try:
            response = client.chat.completions.create(model=config.LLM_MODEL, messages=messages, **settings)  # ask the LLM
            usage = getattr(response, "usage", None)  # token counts, if the provider sends them
            if usage is not None:
                print("   LLM tokens - prompt:", getattr(usage, "prompt_tokens", "?"), "| completion:", getattr(usage, "completion_tokens", "?"))
            return response.choices[0].message.content or ""  # the answer text
        except Exception as error:  # something went wrong
            last_error = error  # remember it
            print("   LLM attempt", attempt + 1, "failed:", str(error)[:300])  # show the problem
            message = str(error).lower()  # lowercase text of the error
            if "per day" in message or "perday" in message:  # the daily limit is used up, more retries would only waste requests
                raise DailyQuotaError("The daily request limit of this model is used up. Wait for the reset (the provider message says when) or choose another model in .env. Provider message: " + str(error)[:200])
            if "temperature" in message and "temperature" in settings:  # the model refuses this setting
                del settings["temperature"]  # try again without it
            elif ("reasoning" in message or "thinking" in message) and "reasoning_effort" in settings:  # the model refuses this setting
                del settings["reasoning_effort"]  # try again without it
            else:
                time.sleep(15 * (attempt + 1))  # wait a little longer after every failure
    raise RuntimeError("The LLM call failed 4 times. Last error: " + str(last_error))


def answer_question(question, store, mode="hybrid", top_k=5, domain="both", use_llm=True):
    # the whole pipeline: retrieve -> (maybe abstain) -> LLM -> check citations
    retrieval = search(question, store, mode, top_k, domain)  # find the best chunks
    results = retrieval["results"]  # the chunks
    top_score = retrieval["top_vector_score"]  # best similarity score

    output = {
        "question": question,
        "results": results,
        "top_vector_score": top_score,
        "retrieval_ms": retrieval["retrieval_ms"],
        "answer": "",
        "abstained": False,
        "abstain_reason": "",
        "llm_used": False,
        "llm_seconds": 0.0,
        "cited_ids": [],
        "invalid_ids": [],
        "citation_ok": None,
        "retries": 0,
    }  # the result dictionary we return

    if len(results) == 0 or top_score < config.ABSTAIN_THRESHOLD:  # nothing similar enough was found
        output["abstained"] = True  # we refuse to answer
        output["abstain_reason"] = "low_retrieval_score"  # why
        output["answer"] = "The indexed documentation does not seem to contain a reliable answer to this question."
        return output  # stop here, no LLM call needed

    if not use_llm or not llm_is_configured():  # no LLM available
        output["answer"] = "No LLM is configured, so only the retrieved passages are shown (see the list below)."
        return output  # stop here

    user_prompt = build_user_prompt(question[:3000], results)  # build the message for the LLM
    answer_text = ""  # the answer we will keep
    for attempt in range(2):  # we ask a second time if the first answer has no citations
        if attempt == 0:  # first try
            prompt = user_prompt  # the normal message
        else:  # second try
            prompt = user_prompt + RETRY_NOTE  # add a note that asks for citations
            output["retries"] = 1  # remember that we had to ask again
        start = time.time()  # start the timer
        answer_text = call_llm(prompt)  # ask the LLM
        output["llm_seconds"] = output["llm_seconds"] + (time.time() - start)  # add the time of this call
        output["llm_used"] = True  # we did call the LLM
        if answer_text.strip().startswith("INSUFFICIENT_EVIDENCE"):  # the LLM refuses, no citations needed
            break  # stop asking
        cited_ids, invalid_ids = find_cited_ids(answer_text, len(results))  # look for [1] [2] numbers
        if len(cited_ids) > 0:  # the answer has at least one valid citation
            break  # good enough, stop asking

    if answer_text.strip().startswith("INSUFFICIENT_EVIDENCE"):  # the LLM says the passages are not enough
        output["abstained"] = True  # count it as abstaining
        output["abstain_reason"] = "model_said_insufficient"  # why
        explanation = answer_text.strip()[len("INSUFFICIENT_EVIDENCE"):].strip()  # the sentence after the keyword
        output["answer"] = "Not enough evidence in the indexed documentation. " + explanation
        return output  # stop here

    output["answer"] = answer_text  # keep the answer
    cited_ids, invalid_ids = find_cited_ids(answer_text, len(results))  # check the [1] [2] numbers
    output["cited_ids"] = cited_ids  # citations that point to real passages
    output["invalid_ids"] = invalid_ids  # citations that point nowhere
    output["citation_ok"] = len(cited_ids) > 0 and len(invalid_ids) == 0  # True when there is at least one citation and all are valid
    return output


def log_query(question, domain, mode, output):
    # adds one line to logs/queries.csv (used by the monitoring section of the app)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)  # create the logs folder if needed
    log_file = config.LOGS_DIR / "queries.csv"  # the log file
    is_new_file = not log_file.exists()  # we write the header only for a new file
    if config.LOG_QUESTION_TEXT:  # store the beginning of the question
        preview = " ".join(question.split())[:80]  # one line, at most 80 characters
    else:  # store nothing about the content
        preview = ""
    with open(log_file, "a", newline="", encoding="utf-8") as f:  # open for appending
        writer = csv.writer(f)  # csv writer
        if is_new_file:  # first line of a new file
            writer.writerow(["timestamp", "question_preview", "question_chars", "domain", "mode", "top_vector_score",
                             "abstained", "abstain_reason", "retrieval_ms", "llm_seconds", "n_cited", "citation_ok"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            preview,
            len(question),
            domain,
            mode,
            round(output["top_vector_score"], 4),
            output["abstained"],
            output["abstain_reason"],
            round(output["retrieval_ms"], 1),
            round(output["llm_seconds"], 2),
            len(output["cited_ids"]),
            output["citation_ok"],
        ])  # the values of this query
