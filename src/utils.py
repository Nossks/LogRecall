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