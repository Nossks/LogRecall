import re
import pandas as pd
import os
from src.utils import mask_log
from dataclasses import dataclass

@dataclass
class data_prep_config:
    input_file = "data/HDFS.log"
    output_file = "artifacts/HDFS_parsed.csv"


class data_prep:
    def __init__(self):
        self.config = data_prep_config()

    def get_id(self, x):
        m = re.search(r"blk_", x)
        if m is None:
            return "unknown"
        if m is None:
            return None
        st=m.start()
        en = x.find(" ",st)
        if en==-1:
            en = len(x)
        return x[st:en] 

    def process_in_chunks(self, chunk_size=100000):
        chunk_list = []
        
        with open(self.config.input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                raw_line = line.strip()
                blk_id = self.get_id(raw_line)
                template = mask_log(raw_line)

                chunk_list.append({
                    'raw_log':raw_line,
                    "template_log":template,
                    "block_id":blk_id
                })
                
                if (i + 1) % chunk_size == 0:
                    df = pd.DataFrame(chunk_list)
                    file_exists = os.path.exists(self.config.output_file)
                    df.to_csv(self.config.output_file, mode='a', header=not file_exists, index=False)
                    chunk_list.clear() 
            
            if chunk_list:
                df = pd.DataFrame(chunk_list)
                file_exists = os.path.exists(self.config.output_file)
                df.to_csv(self.config.output_file, mode='a', header=not file_exists, index=False)
                chunk_list.clear()
