import os
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from prompt_template import build_prompt
from retrieval import retrieve_top_k
from schemas import AskResponse

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]

def _mock_llm_enabled() -> bool:

    return os.environ.get("MOCK_LLM", "1") != "0"

class AssistantState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_chunks: Optional[List[dict]]
    answer: Optional[str]
    sources: Optional[List[str]]
    confidence: Optional[float]

def classify_intent(state: AssistantState) -> AssistantState:
    query_lower = state["query"].lower()

    if _mock_llm_enabled():
        is_policy = any(kw in query_lower for kw in POLICY_KEYWORDS)
        intent = "policy_question" if is_policy else "general_question"
    else:
        intent = _llm_classify_intent(state["query"])

    return {**state, "intent": intent}

def _llm_classify_intent(query: str) -> str:
    from llm_client import call_llm

    prompt = (
        "Classify the following customer question as exactly one word, either "
        "'policy_question' (about Zepto's delivery/returns/membership/tracking/"
        "cancellation/gift cards/support hours) or 'general_question' (anything else).\n"
        f"Question: {query}\nAnswer with exactly one of: policy_question, general_question."
    )
    result = call_llm(prompt).strip().lower()
    return "policy_question" if "policy" in result else "general_question"

def retrieve_and_answer(state: AssistantState) -> AssistantState:

    hits = retrieve_top_k(state["query"], k=3)
    source_ids = [h["id"] for h in hits]

    if _mock_llm_enabled():
        top_chunk_snippet = hits[0]["text"][:200] if hits else ""
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence = 1.0
    else:
        from llm_client import call_llm

        context = "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)
        prompt = build_prompt(question=state["query"], context=context)
        answer = call_llm(prompt).strip()
        confidence = 0.85

    return {
        **state,
        "retrieved_chunks": hits,
        "answer": answer,
        "sources": source_ids,
        "confidence": confidence,
    }

def direct_answer(state: AssistantState) -> AssistantState:
    if _mock_llm_enabled():
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        from llm_client import call_llm

        answer = call_llm(state["query"]).strip()
        confidence = 0.7

    return {**state, "retrieved_chunks": [], "answer": answer, "sources": [], "confidence": confidence}

def route_by_intent(state: AssistantState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

def build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()

_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph

def run_assistant(query: str) -> AskResponse:
    graph = get_graph()
    final_state = graph.invoke({
        "query": query,
        "intent": None,
        "retrieved_chunks": None,
        "answer": None,
        "sources": None,
        "confidence": None,
    })

    return AskResponse(
        answer=final_state["answer"],
        sources=final_state["sources"] or [],
        confidence=final_state["confidence"],
    )
