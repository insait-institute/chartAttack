import logging
import json
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def read_json(file_name: str):
    with open(file_name, 'r') as file:
        json_data = json.load(file)
    return json_data

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)