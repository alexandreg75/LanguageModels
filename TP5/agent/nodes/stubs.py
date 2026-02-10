# TP5/agent/nodes/stubs.py
from TP5.agent.logger import log_event
from TP5.agent.state import AgentState


def stub_reply(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_reply"})
    state.draft_v1 = (
        "Bonjour,\n\n"
        "Merci pour votre message. Je reviens vers vous avec une réponse dès que possible.\n\n"
        "Cordialement,"
    )
    log_event(state.run_id, "node_end", {"node": "stub_reply", "status": "ok"})
    return state


def stub_ask_clarification(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_ask_clarification"})
    state.draft_v1 = (
        "Bonjour,\n\n"
        "Je peux vous aider, mais j’ai besoin de précisions :\n"
        "1) Pouvez-vous confirmer le contexte exact (UE / année / filière) ?\n"
        "2) Pouvez-vous partager le passage ou le document de référence concerné ?\n\n"
        "Merci !"
    )
    log_event(state.run_id, "node_end", {"node": "stub_ask_clarification", "status": "ok"})
    return state


def stub_escalate(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_escalate"})
    state.actions.append({
        "type": "handoff_human",
        "summary": "Demande nécessitant validation/traitement par un humain (service concerné).",
    })
    log_event(state.run_id, "node_end", {"node": "stub_escalate", "status": "ok"})
    return state


def stub_ignore(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_ignore"})
    state.actions.append({
        "type": "ignore",
        "reason": "Email hors périmètre / information non actionnable pour l’assistant.",
    })
    log_event(state.run_id, "node_end", {"node": "stub_ignore", "status": "ok"})
    return state
