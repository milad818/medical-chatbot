import os
from pathlib import Path
import logging



logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# list of files to be created
file_list = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "test/trials.ipynb",
    "app.py",
    "store_index.py",
    "static/.gitkeep",
    "static/style.css",
    "templates/chat.html",
    "test.py",
    "images"
]


for filepath in file_list:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists!")

