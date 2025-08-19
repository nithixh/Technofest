import json
import os

DB_PATH = "user_db.json"

def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return []

def save_database(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)
