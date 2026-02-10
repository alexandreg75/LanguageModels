# TP5/agent/nodes/classify_email.py
import json
import re
from langchain_ollama import ChatOllama

from TP5.agent.logger import log_event
from TP5.agent.prompts import ROUTER_PROMPT
from TP5.agent.state import AgentState, Decision

# Mets le port de ton Ollama
PORT = "11434"

# Mets un modèle QUE TU AS dans `ollama list`
LLM_MODEL = "mistral:instruct"


REPAIR_PROMPT = """\
SYSTEM:
Tu es un correcteur de JSON. Tu ne modifies pas la sémantique.
Tu transforms l'output en JSON strict conforme au schéma.

USER:
Schéma attendu (clés obligatoires) :
{ "intent": "...", "category":"...", "priority":1, "risk_level":"...", "needs_retrieval":true, "retrieval_query":"...", "rationale":"..." }

Output invalide:
<<<{raw}>>>

Retourne UNIQUEMENT le JSON corrigé.
"""


def call_llm(prompt: str) -> str:
    # Important: limiter la sortie + rendre la génération stable
    llm = ChatOllama(
        base_url=f"http://127.0.0.1:{PORT}",
        model=LLM_MODEL,
        temperature=0.0,
        # options -> transmis à l'API Ollama
        options={
            "num_predict": 220,   # limite max tokens générés (JSON court)
            "top_p": 0.9,
        },
        # timeout côté client (évite de bloquer à l'infini)
        timeout=90,
    )
    resp = llm.invoke(prompt)
    txt = resp.content.strip()
    # Nettoyage si le modèle renvoie des balises <think>
    txt = re.sub(r"<think>.*?</think>\s*", "", txt, flags=re.DOTALL).strip()
    return txt


def parse_and_validate(raw: str) -> Decision:
    data = json.loads(raw)
    return Decision(**data)


def classify_email(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "classify_email", "email_id": state.email_id})

    prompt = ROUTER_PROMPT.format(subject=state.subject, sender=state.sender, body=state.body)

    low = (state.body or "").lower()
    if any(x in low for x in ["ignore previous", "system:", "tool", "call", "exfiltrate"]):
        state.decision = Decision(
            intent="escalate",
            category=state.decision.category,
            priority=1,
            risk_level="high",
            needs_retrieval=False,
            retrieval_query="",
            rationale="Suspicion de prompt injection."
        )
        log_event(state.run_id, "node_end", {
            "node": "classify_email",
            "status": "ok",
            "decision": state.decision.model_dump(),
            "note": "injection_heuristic_triggered"
        })
        return state

    try:
        raw = call_llm(prompt)
        decision = parse_and_validate(raw)
    except Exception as e:
        # fallback repair
        log_event(state.run_id, "error", {"node": "classify_email", "kind": "parse_or_validation", "msg": str(e)})

        try:
            # 2e essai (réparation)
            raw2 = call_llm(REPAIR_PROMPT.format(raw=raw if "raw" in locals() else ""))
            decision = parse_and_validate(raw2)
        except Exception as e2:
            # safe fallback: décision minimale
            state.add_error(f"classify_email failed: {e2}")
            decision = Decision(
                intent="ask_clarification",
                category="other",
                priority=3,
                risk_level="med",
                needs_retrieval=False,
                retrieval_query="",
                rationale="Sortie LLM invalide; besoin de clarification.",
            )

    state.decision = decision

    log_event(
        state.run_id,
        "node_end",
        {"node": "classify_email", "status": "ok", "decision": decision.model_dump()},
    )
    return state
