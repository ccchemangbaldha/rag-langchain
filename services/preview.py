# services/preview.py
import uuid
from parser.file_intake import parse_file

def preview_files(paths):
    results = []
    for p in paths:
        items = parse_file(p)
        results.extend(items)
    return results

def merge_files(paths):
    items = preview_files(paths)
    merged = [it["content"] for it in items if it.get("content")]
    full_text = "\n".join(merged).strip()
    return {
        "name": f"merged-{uuid.uuid4()}.txt",
        "content": full_text,
        "files_merged": [it["filename"] for it in items]
    }
