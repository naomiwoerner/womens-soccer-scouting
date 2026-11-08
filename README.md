# womens-soccer-scouting
Streamlit app that turns StatsBomb data into LLM-written scouting reports

## Database

The SQLite database `data/scouting.db` is included in this repository, so you can use it without rebuilding it.

`scripts/build_db.py` recreates it from the raw StatsBomb files in `data/raw/`. Those files are not in the repository, so download them first. From the project root, with the packages in `requirements.txt` installed, run:

```
python -m src.data_loader
python scripts/build_db.py
```
