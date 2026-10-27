# fetch_docs.py - downloads the documentation pages listed in data/sources.csv
# run it with:  python src/fetch_docs.py
import csv  # to read sources.csv
import time  # to pause between downloads (be polite to the websites)
import requests  # to download web pages
import config  # our settings

config.RAW_DIR.mkdir(parents=True, exist_ok=True)  # create data/raw if it does not exist

headers = {"User-Agent": "DevResolve-student-project (RAG learning project, polite downloader)"}  # tell the website who we are

with open(config.SOURCES_FILE, newline="", encoding="utf-8") as f:  # open the list of pages
    rows = list(csv.DictReader(f))  # read all rows (each row has a domain and a url)

print("Pages in the list:", len(rows))  # show how many pages we plan to download

downloaded = 0  # counter for new downloads
skipped = 0  # counter for pages we already have
failed = []  # list of (url, reason) for pages that did not download

for row in rows:  # go through every page
    url = row["url"].strip()  # the page address
    file_name = url.replace("https://", "").replace("/", "_")  # turn the address into a safe file name
    file_path = config.RAW_DIR / file_name  # where we save the page
    if file_path.exists():  # we downloaded this page before
        skipped = skipped + 1  # count it
        continue  # go to the next page
    try:
        response = requests.get(url, headers=headers, timeout=30)  # download the page
    except requests.RequestException as error:  # network problem
        failed.append((url, str(error)))  # remember the problem
        continue  # go to the next page
    if response.status_code != 200:  # the website did not return the page
        failed.append((url, "HTTP " + str(response.status_code)))  # remember the problem
        continue  # go to the next page
    html = response.content.decode("utf-8", errors="replace")  # read the bytes as UTF-8 text
    file_path.write_text(html, encoding="utf-8")  # save the page on disk
    downloaded = downloaded + 1  # count it
    print("Downloaded:", url)  # progress message
    time.sleep(1)  # wait one second before the next download

print("")  # empty line
print("Downloaded now:", downloaded)  # summary
print("Already had:", skipped)  # summary
print("Failed:", len(failed))  # summary
for url, reason in failed:  # list every failure
    print("  FAILED:", url, "->", reason)  # show the failed page and why
