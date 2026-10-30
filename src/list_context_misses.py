import pandas as pd  # to read the CSV file

df = pd.read_csv("experiments/answers.csv", encoding="utf-8-sig")  # read the answers file
answerable = df["is_answerable"].astype(str).str.lower() == "true"  # question has an answer in the corpus
answered = df["abstained"].astype(str).str.lower() == "false"  # the system gave an answer
no_gold = df["gold_in_context"].astype(str).str.lower() == "false"  # the right passage was not retrieved
rows = df[answerable & answered & no_gold]  # all three at once
print(rows[["question_id", "question", "cited_urls"]].to_string(index=False))  # show the rows