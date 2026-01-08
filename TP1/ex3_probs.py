import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- Utils ---
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sentence_logp_and_ppl(model, tokenizer, sentence: str):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab)

    log_probs = torch.log_softmax(logits, dim=-1)
    input_ids = inputs["input_ids"][0]

    total_logp = 0.0
    n = 0
    # On ignore t=0 (pas de contexte avant le premier token)
    for t in range(1, len(input_ids)):
        tok_id = input_ids[t].item()
        lp = log_probs[0, t - 1, tok_id].item()
        total_logp += lp
        n += 1

    avg_neg_logp = - (total_logp / n)
    ppl = math.exp(avg_neg_logp)

    return total_logp, avg_neg_logp, ppl, logits, inputs


def main():
    set_seed(42)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 4.a / 4.b
    phrase = "Artificial intelligence is fascinating."
    total_logp, avg_neg_logp, ppl, logits, inputs = sentence_logp_and_ppl(model, tokenizer, phrase)

    print("=" * 80)
    print("4.a — Probabilités conditionnelles par token")
    print("=" * 80)
    input_ids = inputs["input_ids"][0]

    # softmax pour avoir des probabilités
    probs = torch.softmax(logits, dim=-1)

    # Afficher token et proba conditionnelle P(token_t | tokens_<t)
    # Alignement: token t est prédit à partir de la position t-1 dans logits
    for t in range(1, len(input_ids)):
        tok_id = input_ids[t].item()
        p = probs[0, t - 1, tok_id].item()
        tok_txt = tokenizer.decode([tok_id])
        print(t, repr(tok_txt), f"{p:.3e}")

    print()
    print("=" * 80)
    print("4.b — Log-proba totale + perplexité")
    print("=" * 80)
    print("Phrase:", repr(phrase))
    print("total_logp:", total_logp)
    print("avg_neg_logp:", avg_neg_logp)
    print("perplexity:", ppl)

    # 4.c — comparaison deux phrases anglaises
    s1 = "Artificial intelligence is fascinating."
    s2 = "Artificial fascinating intelligence is."
    tlp1, anlp1, ppl1, _, _ = sentence_logp_and_ppl(model, tokenizer, s1)
    tlp2, anlp2, ppl2, _, _ = sentence_logp_and_ppl(model, tokenizer, s2)

    print()
    print("=" * 80)
    print("4.c — Comparaison phrases anglaises")
    print("=" * 80)
    print("S1:", repr(s1))
    print("  total_logp:", tlp1, "perplexity:", ppl1)
    print("S2:", repr(s2))
    print("  total_logp:", tlp2, "perplexity:", ppl2)

    # 4.d — phrase française
    fr = "L'intelligence artificielle est fascinante."
    tlp_fr, anlp_fr, ppl_fr, _, _ = sentence_logp_and_ppl(model, tokenizer, fr)

    print()
    print("=" * 80)
    print("4.d — Phrase française")
    print("=" * 80)
    print("FR:", repr(fr))
    print("  total_logp:", tlp_fr, "perplexity:", ppl_fr)

    # 4.e — Top-10 prochain token
    prefix = "Artificial intelligence is"
    inp = tokenizer(prefix, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
        logits2 = out.logits  # (1, seq_len, vocab)

    # dernier index temporel = seq_len - 1, donne la distribution du prochain token
    last_logits = logits2[0, -1, :]
    last_probs = torch.softmax(last_logits, dim=-1)

    topk = 10
    vals, idx = torch.topk(last_probs, k=topk)

    print()
    print("=" * 80)
    print("4.e — Top-10 tokens suivants pour le préfixe")
    print("=" * 80)
    print("Prefix:", repr(prefix))
    for p, tid in zip(vals.tolist(), idx.tolist()):
        print(repr(tokenizer.decode([tid])), f"{p:.3e}")


if __name__ == "__main__":
    main()
