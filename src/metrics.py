# metrics.py - simple functions that measure retrieval quality
import numpy as np  # for averages and random numbers


def is_hit(chunk, question):
    # a retrieved chunk is a "hit" when it comes from a gold page
    # AND (if the question lists gold keywords) it contains at least one of them
    if chunk["source_url"] not in question["gold_urls"]:  # wrong page
        return False
    keywords = question.get("gold_keywords", [])  # words that must appear in the chunk
    if len(keywords) == 0:  # no keyword rule, the right page is enough
        return True
    text = (chunk["section"] + " " + chunk["text"]).lower()  # text we search in
    for keyword in keywords:  # check every keyword
        if keyword.lower() in text:  # found one
            return True
    return False  # none of the keywords was found


def first_hit_rank(results, question):
    # returns 1 if the first result is a hit, 2 if the second is the first hit, and so on
    # returns None if no result is a hit
    for position, chunk in enumerate(results):  # results are sorted best first
        if is_hit(chunk, question):  # found the first hit
            return position + 1  # ranks start at 1
    return None  # no hit at all


def recall_at_k(ranks, k):
    # ranks is a list with one rank (or None) per question
    # recall@k = share of questions that have a hit within the top k results
    if len(ranks) == 0:  # no questions
        return 0.0
    hits = 0  # counter
    for rank in ranks:  # look at every question
        if rank is not None and rank <= k:  # hit inside the top k
            hits = hits + 1
    return hits / len(ranks)  # share between 0 and 1


def mean_reciprocal_rank(ranks):
    # MRR = average of 1/rank, where a question without a hit counts as 0
    if len(ranks) == 0:  # no questions
        return 0.0
    total = 0.0  # running sum
    for rank in ranks:  # every question
        if rank is not None:  # there was a hit
            total = total + 1.0 / rank  # a hit at rank 1 gives 1.0, at rank 2 gives 0.5
    return total / len(ranks)  # average


def bootstrap_ci(values, n_rounds=1000, seed=42):
    # a rough 95% range for an average (values are 0 or 1 per question)
    # we resample the questions with replacement many times and look at the spread
    if len(values) == 0:  # no data
        return 0.0, 0.0
    rng = np.random.default_rng(seed)  # random generator with a fixed seed so results repeat
    values = np.array(values, dtype="float64")  # list -> array
    averages = []  # one average per resample
    for _ in range(n_rounds):  # repeat many times
        sample = rng.choice(values, size=len(values), replace=True)  # draw questions with replacement
        averages.append(sample.mean())  # store the average of this resample
    low = float(np.percentile(averages, 2.5))  # lower end of the range
    high = float(np.percentile(averages, 97.5))  # upper end of the range
    return low, high
