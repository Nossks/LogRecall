import pandas as pd
import sqlite3
from dataclasses import dataclass

@dataclass
class sql_db_config:
    db_path = "artifacts/logrecall_metadata.db"
    labels_csv = "data/anomaly_label.csv"
    parsed_logs_csv = "artifacts/HDFS_parsed.csv"
    dict_path = "artifacts/template_dictionary.csv"

class sql_db:
    def __init__(self):
        self.config = sql_db_config()

    def build_sqlite_storage(self):
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS block_labels (
                blk_id TEXT PRIMARY KEY,
                label TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                blk_id TEXT,
                template_id INTEGER,
                raw_log TEXT
            )
        ''')
        
        labels_df = pd.read_csv(self.config.labels_csv)
        labels_df = labels_df.rename(columns={'BlockId': 'blk_id', 'Label': 'label'})
        labels_df.to_sql('block_labels', conn, if_exists='append', index=False)

        
        template_dict = {}
        current_template_id = 1
        
        chunk_size = 100000
        rows_inserted = 0
        
        for chunk in pd.read_csv(self.config.parsed_logs_csv, chunksize=chunk_size):
            chunk = chunk.rename(columns={'block_id': 'blk_id'})
            records_to_insert = []
            
            for _, row in chunk.iterrows():
                template_text = row['template_log']
                
                if template_text not in template_dict:
                    template_dict[template_text] = current_template_id
                    current_template_id += 1
                    
                t_id = template_dict[template_text]
                
                records_to_insert.append((row['blk_id'], t_id, row['raw_log']))
                
            cursor.executemany('''
                INSERT INTO logs (blk_id, template_id, raw_log) 
                VALUES (?, ?, ?)
            ''', records_to_insert)
            
            rows_inserted += len(chunk)
            print(f"Inserted {rows_inserted} physical logs...")
        
        conn.commit()

        cursor.execute('CREATE INDEX idx_blk_id ON logs (blk_id);')
        cursor.execute('CREATE INDEX idx_template_id ON logs (template_id);')
        conn.commit()
        
        templates_df = pd.DataFrame(list(template_dict.items()), columns=['template_log', 'template_id'])
        templates_df.to_csv(self.config.dict_path, index=False)
        
        conn.close()