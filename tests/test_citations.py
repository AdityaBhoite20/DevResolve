# test_citations.py - checks the citation finder
from citations import find_cited_ids


def test_finds_valid_and_invalid_citations():
    answer = "The scaler learns the mean [1]. Use a Pipeline [2, 3]. This one is made up [9]."
    cited, invalid = find_cited_ids(answer, 5)  # we gave 5 passages
    assert cited == [1, 2, 3]  # valid numbers
    assert invalid == [9]  # passage 9 does not exist


def test_code_indexing_is_not_a_citation():
    cited, invalid = find_cited_ids("Use items[0] to get the first item [1].", 3)  # [0] is code
    assert cited == [1]
    assert invalid == []


def test_no_citations():
    cited, invalid = find_cited_ids("An answer without any citation.", 3)
    assert cited == [] and invalid == []
