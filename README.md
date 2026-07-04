# template_repo
---
## 📁 Directory Structure Explained

Here is a breakdown of what each file and folder is responsible for:

```
├── README.md           # Project documentation
├── artifacts/            # Stores output files like trained models (.pkl) or preprocessors (.pkl)
├── notebooks/            # Contains Jupyter notebooks for experimentation (EDA, model prototyping)
├── requirements.txt      # Lists all Python libraries required for the project
├── setup.py              # Makes the 'src' directory installable as a local Python package
└── src/                  # The main source code for the project
    ├── __init__.py       # Makes 'src' a Python package
    ├── components/       # Contains individual modules for each step of the ML pipeline
    │   ├── __init__.py
    │   ├── data_ingestion.py     # Script to get and save the data
    │   ├── data_transformation.py  # Script for all data preprocessing and feature engineering
    │   └── model_trainer.py      # Script to train and save the model
    │
    ├── pipeline/         # Contains scripts that orchestrate the components
    │   ├── __init__.py
    │   ├── predict_pipeline.py   # Script to load the trained model and make new predictions
    │   └── train_pipeline.py     # Script to run the full training workflow (ingestion -> transformation -> training)
    │
    ├── exception.py      # Custom exception handling for detailed error messages
    ├── logger.py         # Configures a custom logger to save logs to a file
    └── utils.py          # Utility functions (e.g., save_object, load_object, evaluate_model)
```

### Key Files

* **`setup.py`**: This file is crucial. It allows you to run `pip install .` in your terminal, which packages your entire `src` directory. This lets you import your code from anywhere using `from src.components.data_ingestion import DataIngestion`.
* **`src/logger.py`**: All `logging.info("message")` calls will be written to a log file in the `logs/` directory (which this script creates).
* **`src/exception.py`**: Allows you to `raise CustomException(e, sys)` to get a perfectly formatted error message showing the exact file and line number where the error occurred.
* **`src/utils.py`**: A central place for helper functions. A common function here is `save_object()`, which is used in `data_transformation.py` to save the preprocessor and in `model_trainer.py` to save the trained model.
* **`artifacts/`**: This directory is *not* for code. It's where your `train_pipeline.py` will *save* its outputs (e.g., `model.pkl`, `preprocessor.pkl`).
