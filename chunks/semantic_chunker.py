# chunks/semantic_chunker.py
import spacy
import re

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import en_core_web_sm
    nlp = en_core_web_sm.load()

MAX_TOKENS = 300      
OVERLAP_TOKENS = 80   

def count_tokens(text: str):
    """
    Fast whitespace-based token counting.
    """
    return len(text.split())

def clean_text(text: str) -> str:
    """
    CRITICAL STEP: Cleans PDF noise before processing.
    """
    if not text:
        return ""
    
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'--- PAGE \d+ ---', '', text)
    
    return text.strip()

def create_smart_chunks(content: str, source_files: list):
    """
    Splits text into clean, sliding windows of sentences.
    """
    cleaned_content = clean_text(content)
    
    nlp.max_length = len(cleaned_content) + 100000
    doc = nlp(cleaned_content)
    
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    chunks = []
    chunk_index = 0
    
    current_chunk_sents = []
    current_length = 0
    
    for sentence in sentences:
        sent_len = count_tokens(sentence)
        
        if sent_len < 2:
            continue

        if current_length + sent_len > MAX_TOKENS and current_chunk_sents:
            
            text_block = " ".join(current_chunk_sents)
            chunks.append({
                "chunk_index": chunk_index,
                "text": text_block,
                "tokens": current_length,
                "source_files": source_files
            })
            chunk_index += 1
            
            overlap_buffer = []
            overlap_len = 0
            
            for old_sent in reversed(current_chunk_sents):
                old_len = count_tokens(old_sent)
                if overlap_len + old_len < OVERLAP_TOKENS:
                    overlap_buffer.insert(0, old_sent)
                    overlap_len += old_len
                else:
                    break
            
            current_chunk_sents = overlap_buffer
            current_length = overlap_len

        current_chunk_sents.append(sentence)
        current_length += sent_len

    if current_chunk_sents:
        chunks.append({
            "chunk_index": chunk_index,
            "text": " ".join(current_chunk_sents),
            "tokens": current_length,
            "source_files": source_files
        })

    return chunks

def split_sentences(content): return [] 
def cluster_sentences(s, v, f): return []