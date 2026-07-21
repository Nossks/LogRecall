import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

@dataclass
class faiss_db_config:
    model_path = "artifacts/stage1_encoder.pth"
    dict_path = "artifacts/template_dictionary.csv"
    index_path = "artifacts/logrecall_core.index"
    
class faiss_db:
    def __init__(self):
        self.config = faiss_db_config()
        
    def build_faiss_index(self):
        
        df = pd.read_csv(self.config.dict_path)
        
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2') 
        
        raw_state_dict = torch.load(self.config.model_path, map_location=torch.device('cpu'))
        
        clean_state_dict = {}
        for key, value in raw_state_dict.items():
            new_key = key.replace('model.', '', 1) if key.startswith('model.') else key
            clean_state_dict[new_key] = value
        
        try:
            model[0].auto_model.load_state_dict(clean_state_dict)
            print("Custom AI Model booted and locked.")
        except Exception as e:
            print(f"Injection Failed: {e}")
            return

        
        templates = df['template_log'].tolist()
        vectors = model.encode(templates, batch_size=256, show_progress_bar=True)
        vectors = np.array(vectors).astype('float32') 
        

        dimension = vectors.shape[1] 
        faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        
        template_ids = df['template_id'].values.astype('int64')
        faiss_index.add_with_ids(vectors, template_ids)
        
        faiss.write_index(faiss_index, self.config.index_path)