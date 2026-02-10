# TP5/agent/routing.py
import re
from TP5.agent.state import AgentState

PII_RE = re.compile(
    r"\b("
    r"liste\s+des\s+notes|"
    r"notes\s+des\s+étudiants|"
    r"liste\s+des\s+étudiants|"
    r"relev[eé]\s+de\s+notes|"
    r"bulletin|"
    r"résultats?\s+des\s+étudiants|"
    r"toutes\s+les\s+notes|"
    r"moyennes?\s+des\s+étudiants"
    r")\b",
    re.IGNORECASE,
)

def route(state: AgentState) -> str:
    """
    Routing déterministe (testable).
    Le LLM propose une décision, mais le code applique des garde-fous.
    """

    subject = state.subject or ""
    body = state.body or ""
    text = f"{subject} {body}"

    # 1) Garde-fou sécurité (PII / notes / listes nominatives) -> escalade
    if PII_RE.search(text):
        return "escalate"

    # 2) Si le modèle a marqué risque high -> escalade
    if state.decision.risk_level == "high":
        return "escalate"

    # 3) Sinon, routage normal
    intent = state.decision.intent
    if intent == "reply":
        return "reply"
    if intent == "ask_clarification":
        return "ask_clarification"
    if intent == "escalate":
        return "escalate"
    return "ignore"
