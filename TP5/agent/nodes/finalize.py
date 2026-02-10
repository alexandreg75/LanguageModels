# TP5/agent/nodes/finalize.py
import re
from typing import List

from TP5.agent.logger import log_event
from TP5.agent.state import AgentState

RE_CIT = re.compile(r"\[(doc_\d+)\]")


def _extract_citations(text: str) -> List[str]:
    return sorted(set(RE_CIT.findall(text or "")))


def _has_handoff_action(state: AgentState) -> bool:
    return any(
        (a or {}).get("type") in {"handoff_human", "handoff_packet"}
        for a in (state.actions or [])
    )


def finalize(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "finalize"})

    # ------------------------------------------------------------------
    # PRIORITÉ SÉCURITÉ / COHÉRENCE:
    # Si un noeud a déjà déclenché une escalade via state.actions,
    # on force un rendu final "handoff" (même si decision.intent == "reply").
    # ------------------------------------------------------------------
    if _has_handoff_action(state):
        state.final_kind = "handoff"

        # Normaliser: si on n'a pas déjà un handoff_packet, en créer un.
        if not any((a or {}).get("type") == "handoff_packet" for a in (state.actions or [])):
            summary = (
                f"Sujet: {state.subject}. "
                f"Demande nécessitant validation humaine (cat={state.decision.category}, "
                f"priorité={state.decision.priority}, risque={state.decision.risk_level})."
            )
            state.actions.append(
                {
                    "type": "handoff_packet",
                    "run_id": state.run_id,
                    "email_id": state.email_id,
                    "summary": summary,
                    "evidence_ids": [d.doc_id for d in state.evidence],
                }
            )

        state.final_text = (
            "Bonjour,\n\n"
            "Votre demande nécessite une validation humaine. Je la transmets à la personne / équipe concernée "
            "avec un résumé et les sources disponibles.\n\n"
            "Cordialement,"
        )

        log_event(
            state.run_id,
            "node_end",
            {"node": "finalize", "status": "ok", "final_kind": state.final_kind},
        )
        return state

    # ------------------------------------------------------------------
    # Sinon, logique standard basée sur intent
    # ------------------------------------------------------------------
    intent = state.decision.intent

    if intent == "reply":
        cits = _extract_citations(state.draft_v1)
        state.final_kind = "reply"

        base = (state.draft_v1 or "").strip()
        if not base:
            base = (
                "Bonjour,\n\n"
                "Merci pour votre message. Je reviens vers vous dès que possible avec une réponse étayée.\n\n"
                "Cordialement,"
            )

        if cits:
            state.final_text = base + "\n\nSources: " + " ".join(f"[{c}]" for c in cits)
        else:
            state.final_text = base

    elif intent == "ask_clarification":
        state.final_kind = "clarification"
        base = (state.draft_v1 or "").strip()
        if not base:
            base = (
                "Bonjour,\n\n"
                "Pour pouvoir vous répondre précisément, pourriez-vous préciser :\n"
                "1) le contexte exact (UE/ECUE, année/cursus, etc.) ?\n"
                "2) l’objectif attendu (validation, rattrapage, procédure, etc.) ?\n"
                "3) toute contrainte de délai éventuelle ?\n\n"
                "Merci d’avance.\n\n"
                "Cordialement,"
            )
        state.final_text = base

    elif intent == "escalate":
        state.final_kind = "handoff"

        summary = (
            f"Sujet: {state.subject}. "
            f"Demande classée en escalade (cat={state.decision.category}, priorité={state.decision.priority}, risque={state.decision.risk_level}). "
            f"Besoin: validation humaine / action hors périmètre agent."
        )

        state.actions.append(
            {
                "type": "handoff_packet",
                "run_id": state.run_id,
                "email_id": state.email_id,
                "summary": summary,
                "evidence_ids": [d.doc_id for d in state.evidence],
            }
        )

        state.final_text = (
            "Bonjour,\n\n"
            "Votre demande nécessite une validation humaine. Je la transmets à la personne / équipe concernée "
            "avec un résumé et les sources disponibles.\n\n"
            "Cordialement,"
        )

    else:
        state.final_kind = "ignore"
        state.final_text = (
            "Message ignoré car hors périmètre (informatif / non actionnable) ou ne nécessitant pas de réponse."
        )

    log_event(state.run_id, "node_end", {"node": "finalize", "status": "ok", "final_kind": state.final_kind})
    return state
