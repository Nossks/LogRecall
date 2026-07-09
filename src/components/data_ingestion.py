import pandas as pd
import os
import re
import hashlib
from dataclasses import dataclass

@dataclass
class DataPreprocessConfig:
    org_log_path = 'data/HDFS.log'
    org_anomaly_path = "data/anomaly_label.csv"
    stage1_output_path = "artifacts/stage1_triplet_data.parquet"

class DataPreprocess:
    def __init__(self):
        self.config = DataPreprocessConfig()
        
    def mask_log(self, raw_log: str) -> str:
        cleaned = re.sub(r'^\s*\d+\s+\d+\s+\d+\s+[A-Z]+\s+[^:]+:\s*', '', raw_log)
        cleaned = re.sub(r'blk_-?\d+', '<BLOCK_ID>', cleaned)
        cleaned = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', cleaned)
        cleaned = re.sub(r'\b\d+\b', '<NUM>', cleaned)
        return cleaned.strip()

    def generate_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

    def initiate_dp(self):
        os.makedirs('artifacts', exist_ok=True)
        processed_records = []
        with open(self.config.org_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                masked_text = self.mask_log(line)
                hash_id = self.generate_hash(masked_text)
                
                processed_records.append({
                    'masked_log': masked_text,
                    'hash_id': hash_id
                })
        df = pd.DataFrame(processed_records)
        df = df.drop_duplicates().reset_index(drop=True)
        df.to_parquet(self.config.stage1_output_path, index=False)

# if __name__ == "__main__":
#     obj = DataPreprocess()
#     obj.initiate_dp()