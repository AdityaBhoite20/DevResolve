# citations.py - finds the citation numbers like [1] or [2, 3] inside an LLM answer
import re  # to find the numbers inside square brackets


def find_cited_ids(answer_text, n_sources):
    # n_sources is how many passages we gave to the LLM (numbered 1 to n_sources)
    groups = re.findall(r"\[([\d,\s]+)\]", answer_text)  # every [...] that holds only digits, commas and spaces
    cited_ids = []  # valid citation numbers
    invalid_ids = []  # numbers that do not match any passage
    for group in groups:  # for example "1, 2"
        for part in group.split(","):  # split into "1" and " 2"
            part = part.strip()  # remove spaces
            if not part.isdigit():  # skip empty parts
                continue
            number = int(part)  # convert text to a number
            if number < 1:  # [0] is almost always code like items[0], not a citation
                continue
            if number <= n_sources:  # this passage really exists
                if number not in cited_ids:  # keep every number only once
                    cited_ids.append(number)
            else:  # the LLM cited a passage that was never given to it
                if number not in invalid_ids:
                    invalid_ids.append(number)
    cited_ids.sort()  # smallest first
    invalid_ids.sort()
    return cited_ids, invalid_ids
