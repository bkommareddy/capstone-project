# Module 1 — Data Pipeline

baseline rate, and loads it into a normalized SQLite database that's then queried with
both SQL and pandas.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python scrape_and_load.py
python queries.py
```

Requires internet access to books.toscrape.com. Produces, in this folder:

- `books_raw.csv` — raw scraped rows before cleaning
- `books_clean.csv` — cleaned rows actually loaded into the DB
- `zepto_books.db` — the SQLite database (2-table normalized schema)

## SQL queries (`queries.py`)

Five queries are run and their output printed, collectively covering
`SELECT`/`WHERE`, `ORDER BY`/`LIMIT`, `DISTINCT`, `BETWEEN`, and a `JOIN`
between `books` and `categories`. Two of the query results are then read back
into pandas via `pd.read_sql`, and the JOIN query is separately reproduced
with `pd.merge` on the in-memory DataFrames — the script prints both results
and confirms they're equal.
