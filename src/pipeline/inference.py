import torch
import faiss
import sqlite3
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from src.utils import mask_log

@dataclass
class InferenceConfig:
    model_path = "artifacts/stage1_encoder.pth"
    faiss_path = "artifacts/logrecall_core.index"
    sql_path = "artifacts/logrecall_metadata.db"
    dict_path = "artifacts/template_dictionary.csv"

class Inference:
    def __init__(self):
        self.config = InferenceConfig()
        self.sql_conn = sqlite3.connect(self.config.sql_path, check_same_thread=False)       
        self.faiss_index = faiss.read_index(self.config.faiss_path)      
        self.model = SentenceTransformer('all-MiniLM-L12-v2') 
        raw_state_dict = torch.load(self.config.model_path, map_location=torch.device('cpu'))      
        clean_state_dict = {}
        for key, value in raw_state_dict.items():
            new_key = key.replace('model.', '', 1) if key.startswith('model.') else key
            clean_state_dict[new_key] = value         
        self.model[0].auto_model.load_state_dict(clean_state_dict)

    def get_embedding(self,logs):
        lines = [line.strip() for line in logs.split('\n') if line.strip()]
        if not lines:
            return None
        masked_lines = [mask_log(line) for line in lines]
        vectors = self.model.encode(masked_lines)

        return np.array(vectors).astype('float32')
    
    def get_results(self, embeddings, k=2, threshold=0.75):
        if embeddings is None or len(embeddings) == 0:
            return []
            
        distances, indices = self.faiss_index.search(embeddings, k)
        
        matched_template_ids = set()
        
        for row_idx, dist_row in enumerate(distances):
            for col_idx, score in enumerate(dist_row):
                if score >= threshold:
                    template_id = indices[row_idx][col_idx]
                    if template_id != -1:
                        matched_template_ids.add(int(template_id))
        
        if not matched_template_ids:
            return []
            
        cursor = self.sql_conn.cursor()
        placeholders = ','.join('?' * len(matched_template_ids))
        
        query = f"""
            SELECT template_id, COUNT(blk_id) as total_occurrences, MAX(raw_log) as sample_log 
            FROM logs 
            WHERE template_id IN ({placeholders}) 
            GROUP BY template_id
        """

        cursor.execute(query, tuple(matched_template_ids))
        rows = cursor.fetchall()

        return [
            {
                "template_id": r[0], 
                "total_system_occurrences": r[1], 
                "sample_log": r[2]
            } 
            for r in rows
        ]
    
# if __name__ == "__main__":
#     obj = Inference()
#     io="""
#         081111 103015 25 WARN dfs.DataNode$DataXceiver: Got exception while serving a block to a remote client
#         java.io.IOException: Connection reset by peer
#         at sun.nio.ch.FileChannelImpl.transferTo0(Native Method)
#         at sun.nio.ch.FileChannelImpl.transferToDirectly(FileChannelImpl.java:433)
#         at org.apache.hadoop.dfs.DataNode$DataXceiver.readBlock(DataNode.java:736)
# """
#     embeddings = obj.get_embedding(io)
#     print(obj.get_results(embeddings,1))