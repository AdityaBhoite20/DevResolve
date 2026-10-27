# chunk_docs.py - cuts every document section into chunks (small pieces of text used for search)
# run it with:  python src/chunk_docs.py   (it saves data/processed/chunks.jsonl so you can inspect the chunks)
import json  # to read and write JSON lines files
import config  # our settings


def split_long_block(block, chunk_size):
    # one paragraph or code block can be longer than a chunk, so we cut it into smaller pieces
    pieces = []  # the pieces we will return
    while len(block) > chunk_size:  # keep cutting while the block is too long
        cut = block.rfind("\n", 0, chunk_size)  # try to cut at the last line break
        if cut < chunk_size // 2:  # a line break very early in the text is not useful
            cut = block.rfind(" ", 0, chunk_size)  # then try to cut at the last space
        if cut <= 0:  # no good place found
            cut = chunk_size  # cut exactly at the size limit
        pieces.append(block[:cut].strip())  # keep the first part
        block = block[cut:].strip()  # continue with the rest
    pieces.append(block)  # the rest is short enough
    return pieces


def make_chunks(documents, chunk_size, chunk_overlap):
    # documents come from data/processed/documents.jsonl
    chunks = []  # all chunks of all pages
    for doc_number, doc in enumerate(documents):  # every page
        chunk_number = 0  # counts the chunks of this page
        for section in doc["sections"]:  # every section of the page
            pieces = []  # paragraphs and code blocks, cut to a maximum size
            for block in section["blocks"]:  # every paragraph or code block
                for piece in split_long_block(block, chunk_size):  # cut it if it is too long
                    if piece != "":  # ignore empty pieces
                        pieces.append(piece)  # keep the piece

            texts = []  # finished chunk texts of this section
            current = ""  # the chunk we are building right now
            for piece in pieces:  # add the pieces one by one
                if (piece.startswith("exception ") or piece.startswith("class ")) and current != "":  # a new exception or class is described here
                    texts.append(current)  # finish the chunk before it, so every exception or class gets its own chunk
                    current = ""  # and start fresh (no overlap across two different definitions)
                if current != "" and len(current) + len(piece) + 1 > chunk_size:  # the chunk would become too big
                    texts.append(current)  # finish the current chunk
                    if chunk_overlap > 0:  # repeat the end of the old chunk in the new one
                        tail = current[-chunk_overlap:]  # last characters of the old chunk
                        tail = tail.split(" ", 1)[-1]  # drop the first (maybe cut) word
                    else:
                        tail = ""  # no overlap wanted
                    current = tail  # start the new chunk with the overlap
                if current == "":  # the new chunk is empty
                    current = piece  # start it with this piece
                else:
                    current = current + "\n" + piece  # add the piece after a line break
            if current != "":  # the last chunk of the section
                texts.append(current)  # keep it

            for text in texts:  # turn every text into a chunk record
                if len(text) < config.MIN_CHUNK_CHARS:  # too short to be useful
                    continue
                section_url = doc["url"]  # link to the page
                if section["anchor"] != "":  # link to the exact section when we know it
                    section_url = doc["url"] + "#" + section["anchor"]
                chunks.append({
                    "chunk_id": "d" + str(doc_number).zfill(3) + "_c" + str(chunk_number).zfill(3),
                    "source_url": doc["url"],
                    "section_url": section_url,
                    "title": doc["title"],
                    "domain": doc["domain"],
                    "section": section["heading"],
                    "version": doc["version"],
                    "fetched_at": doc["fetched_at"],
                    "text": text,
                    "embed_text": doc["title"] + " | " + section["heading"] + "\n" + text,
                })  # save the chunk with all its metadata
                chunk_number = chunk_number + 1  # next chunk of this page
    return chunks


def load_documents():
    # reads data/processed/documents.jsonl and returns a list of documents
    documents = []  # the list we return
    with open(config.PROCESSED_DIR / "documents.jsonl", encoding="utf-8") as f:  # open the file
        for line in f:  # one JSON object per line
            if line.strip() != "":  # skip empty lines
                documents.append(json.loads(line))  # convert the text to a dictionary
    return documents


if __name__ == "__main__":  # this part runs only when you start the file directly
    documents = load_documents()  # read the parsed pages
    chunks = make_chunks(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)  # cut them into chunks
    with open(config.PROCESSED_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:  # write the chunks
        for chunk in chunks:  # one JSON object per line
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")  # write the line
    lengths = [len(chunk["text"]) for chunk in chunks]  # length of every chunk
    print("Chunk size setting:", config.CHUNK_SIZE, "| overlap:", config.CHUNK_OVERLAP)  # settings used
    print("Pages:", len(documents))  # number of pages
    print("Chunks:", len(chunks))  # number of chunks
    if len(lengths) > 0:  # avoid errors on an empty list
        print("Average chunk length:", round(sum(lengths) / len(lengths)), "characters")  # average
        print("Shortest / longest chunk:", min(lengths), "/", max(lengths))  # extremes
    for domain in ["python", "scikit-learn"]:  # chunks per domain
        count = 0  # counter
        for chunk in chunks:  # check every chunk
            if chunk["domain"] == domain:  # same domain
                count = count + 1  # count it
        print("Chunks in", domain + ":", count)  # show the number
