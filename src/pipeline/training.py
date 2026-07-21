from src.logger import logging
from src.components.Data_Ingestion.faiss_ingestion import faiss_db
from src.components.Data_Ingestion.sql_ingestion import sql_db
from src.components.Data_Ingestion.data_preprocess import DataPreprocess
from src.components.Data_Ingestion.data_prepration import data_prep
from src.components.Model.model_trainer import Trainer

def main():
    logging.info("starting inference pipeline")

    logging.info("starting data preprocessing for model fine tuning")
    preprocess = DataPreprocess()
    preprocess.initiate_dp()

    logging.info("starting model fine tuning")
    model = Trainer(3)
    model.InitiateTraining()

    logging.info("starting data preparation for before db ingestion's")
    prepare = data_prep()
    prepare.process_in_chunks()

    logging.info("starting sql db ingestion")
    sql = sql_db()
    sql.build_sqlite_storage()

    logging.info("starting faiss db ingestion")
    faiss = faiss_db()
    faiss.build_faiss_index()

if __name__ == "__main__":
    main()