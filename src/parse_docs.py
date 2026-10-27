# parse_docs.py - turns the downloaded HTML pages into clean text sections
# run it with:  python src/parse_docs.py
import csv  # to read sources.csv
import json  # to write the documents file
import re  # text patterns
from datetime import datetime  # to record the download date
from bs4 import BeautifulSoup  # reads HTML
import config  # our settings


def html_to_sections(html):
    # turns one HTML page into (title, version, sections)
    # every section is {"heading": ..., "anchor": ..., "blocks": [paragraph, paragraph, code, ...]}
    soup = BeautifulSoup(html, "html.parser")  # parse the HTML text

    title_tag = soup.find("title")  # the <title> tag of the page
    if title_tag is not None:  # the page has a title
        full_title = " ".join(title_tag.get_text().split())  # tidy the spaces
    else:
        full_title = ""  # no title found
    short_title = re.split(r"\s[—–-]\s", full_title)[0]  # cut off the ending like "- scikit-learn 1.x documentation"
    version_match = re.search(r"(?:scikit-learn|Python)\s+(\d+\.\d+(?:\.\d+)?)", full_title)  # look for a version number
    if version_match is not None:  # found one
        version = version_match.group(1)  # for example "1.8.0"
    else:
        version = "unknown"  # no version in the title

    content = None  # the part of the page that holds the real documentation text
    for selector in ["article.bd-article", "div[role=main]", "main", "div.body", "div.document"]:  # try these places in order
        content = soup.select_one(selector)  # look for the place
        if content is not None:  # found it
            break  # stop searching
    if content is None:  # none of the places exist
        content = soup.body  # use the whole page body
    if content is None:  # the page has no body at all
        return short_title, version, []  # nothing to extract

    for tag in content.find_all(["script", "style", "nav", "footer", "aside", "form", "button"]):  # page parts that are not documentation
        tag.extract()  # remove them from the page
    for tag in content.select("a.headerlink, .sphx-glr-thumbcontainer, .sphx-glr-thumbnails, .sphx-glr-footer, .bd-footer-article, .prev-next-area"):  # more noise
        tag.extract()  # remove them too

    codes = []  # list that keeps the code blocks
    for pre in content.find_all("pre"):  # every code block
        codes.append(pre.get_text().strip("\n"))  # keep its text with line breaks
        pre.replace_with("CODEBLOCK" + str(len(codes) - 1) + "ENDCODE")  # leave a marker in its place

    block_tags = ["p", "li", "dt", "dd", "tr", "div", "ul", "ol", "dl", "table", "blockquote", "h5", "h6"]  # tags that start a new paragraph
    for tag in content.find_all(block_tags):  # for each of them
        tag.insert_before("\n\n")  # put an empty line before
        tag.insert_after("\n\n")  # and after, so paragraphs stay separate
    for cell in content.find_all(["td", "th"]):  # table cells
        cell.insert_after(" | ")  # separate cells with a bar

    sections = []  # the sections we will return
    for heading in content.find_all(["h1", "h2", "h3", "h4"]):  # every heading starts a section
        heading_text = " ".join(heading.get_text().split()).rstrip("#¶ ")  # clean heading text
        anchor = ""  # the id used for a link to this section
        if heading.parent is not None and heading.parent.get("id"):  # sphinx puts the id on the parent
            anchor = heading.parent.get("id")  # keep it
        blocks = []  # paragraphs of this section
        for sibling in heading.next_siblings:  # walk through what comes after the heading
            if sibling.name is None:  # plain text (like a line break), skip
                continue
            if sibling.name in ["h1", "h2", "h3", "h4", "section"]:  # next heading or a sub-section starts
                break  # this section ends here
            if sibling.name == "div" and "section" in (sibling.get("class") or []):  # old style sub-section
                break  # this section ends here
            pieces = re.split(r"\n\s*\n", sibling.get_text())  # split the text at empty lines
            for piece in pieces:  # every paragraph
                piece = " ".join(piece.split())  # remove extra spaces and line breaks
                if piece == "":  # nothing left
                    continue
                for part in re.split(r"(CODEBLOCK\d+ENDCODE)", piece):  # separate code markers from normal text
                    part = part.strip()  # remove spaces
                    marker = re.fullmatch(r"CODEBLOCK(\d+)ENDCODE", part)  # is this part a code marker?
                    if marker is not None:  # yes
                        blocks.append(codes[int(marker.group(1))])  # put the real code back
                    elif part != "":  # normal text
                        blocks.append(part)  # keep the paragraph
        if len(blocks) > 0:  # skip empty sections
            sections.append({"heading": heading_text, "anchor": anchor, "blocks": blocks})  # save the section
    return short_title, version, sections


if __name__ == "__main__":  # this part runs only when you start the file directly
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)  # create data/processed if needed

    with open(config.SOURCES_FILE, newline="", encoding="utf-8") as f:  # open the list of pages
        rows = list(csv.DictReader(f))  # read all rows

    documents = []  # one entry per page
    missing = []  # pages that were never downloaded
    thin_pages = []  # pages with very little text (extraction may have failed)

    for row in rows:  # go through every page
        url = row["url"].strip()  # page address
        file_path = config.RAW_DIR / url.replace("https://", "").replace("/", "_")  # file saved by fetch_docs.py
        if not file_path.exists():  # not downloaded
            missing.append(url)  # remember it
            continue
        html = file_path.read_text(encoding="utf-8")  # read the HTML
        title, version, sections = html_to_sections(html)  # extract the text sections
        total_chars = 0  # count the characters of this page
        for section in sections:  # every section
            for block in section["blocks"]:  # every paragraph
                total_chars = total_chars + len(block)  # add its length
        fetched_at = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d")  # day the file was downloaded
        documents.append({
            "url": url,
            "title": title,
            "domain": row["domain"].strip(),
            "version": version,
            "fetched_at": fetched_at,
            "sections": sections,
        })  # save the page
        print(row["domain"], "|", title, "| version", version, "|", len(sections), "sections |", total_chars, "chars")  # progress line
        if total_chars < 200:  # very little text
            thin_pages.append(url)  # remember it

    with open(config.PROCESSED_DIR / "documents.jsonl", "w", encoding="utf-8") as f:  # write the result file
        for document in documents:  # one JSON object per line
            f.write(json.dumps(document, ensure_ascii=False) + "\n")  # write the line

    print("")  # empty line
    print("Pages parsed:", len(documents))  # summary
    print("Pages missing (not downloaded):", len(missing))  # summary
    for url in missing:  # show them
        print("  MISSING:", url)
    print("Pages with very little text (please check):", len(thin_pages))  # summary
    for url in thin_pages:  # show them
        print("  THIN:", url)
