linked modules — a data-engineering pipeline, an analytics pipeline, and a GenAI
support assistant. Each module has its own folder and grading rubric; together they
form a single submission.
| Module | Folder | Marks | What it does |
|---|---|---|---|
| Data Pipeline | [`/data_pipeline`](./data_pipeline) | 25 | Scrapes books.toscrape.com, cleans and converts pricing, loads into a normalized SQLite DB, queries with SQL + pandas |
| Analytics Pipeline | [`/analytics`](./analytics) | 50 | Profiles and cleans the Titanic dataset, tells a visual data story, then builds/tunes/evaluates a full classification + regression modeling pipeline |
| Support Assistant | [`/support_assistant`](./support_assistant) | 25 | RAG-based GenAI assistant over Zepto's own policy docs, orchestrated with LangGraph and served via FastAPI |
## Setup
Each module has its own `requirements.txt`.
```bash
pip install -r requirements.txt
```
No paid services or API keys are required anywhere for the graded baseline of
any module. See each module's README for specifics.
## Running each module end to end
**1. Data Pipeline**
```bash
cd data_pipeline
pip install -r requirements.txt
python scrape_and_load.py
python queries.py
```
**2. Analytics Pipeline**
```bash
cd analytics
pip install -r requirements.txt
python 01_eda.py
python 02_modeling.py
```
**3. Support Assistant**
```bash
cd support_assistant
pip install -r requirements.txt
python ingest.py
uvicorn main:app --host 0.0.0.0 --port 7860
```
Or via Docker:
```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```
