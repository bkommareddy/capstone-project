from fastapi import FastAPI

from graph import run_assistant
from schemas import AskRequest, AskResponse

app = FastAPI(title="Zepto Support Assistant")

@app.get("/")
def root():
    return {"status": "ok", "service": "zepto-support-assistant"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return run_assistant(request.query)
