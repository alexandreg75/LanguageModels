from TP5.agent.logger import log_event
from TP5.agent.state import AgentState


def check_evidence(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "check_evidence"})

    # Heuristique pragmatique: ok si la dernière génération a produit des citations valides
    state.evidence_ok = bool(state.last_draft_had_valid_citations)

    log_event(state.run_id, "node_end", {
        "node": "check_evidence",
        "status": "ok",
        "evidence_ok": state.evidence_ok,
        "last_draft_had_valid_citations": state.last_draft_had_valid_citations,
        "retrieval_attempts": state.budget.retrieval_attempts,
        "max_retrieval_attempts": state.budget.max_retrieval_attempts,
    })
    return state
