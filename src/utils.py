def create_collate_fn(tokenizer, max_len=128):
    def custom_collate(batch):
        anchors, positives, negatives = zip(*batch)
        def tokenize_group(texts):
            return tokenizer(
                list(texts),
                padding=True, 

                truncation=True, 
                max_length=max_len,
                return_tensors="pt" 
            )
        return {
            "anchor": tokenize_group(anchors),
            "positive": tokenize_group(positives),
            "negative": tokenize_group(negatives)
        }
    return custom_collate

import re

def mask_log(raw_log: str) -> str:
    cleaned = re.sub(r'^\s*\d+\s+\d+\s+\d+\s+[A-Z]+\s+[^:]+:\s*', '', raw_log)
    cleaned = re.sub(r'blk_-?\d+', '<BLOCK_ID>', cleaned)
    cleaned = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', cleaned)
    cleaned = re.sub(r'\b\d+\b', '<NUM>', cleaned)
    return cleaned.strip()