import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

SEED = 42
PROMPT = "The future of artificial intelligence is"

MAX_LEN = 50  # longueur max en tokens (attention: inclut les tokens du prompt)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode(tokenizer, ids):
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def main():
    print("SEED (base):", SEED)
    print("PROMPT:", repr(PROMPT))
    print("MAX_LEN:", MAX_LEN)
    print()

    set_seed(SEED)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    inputs = tokenizer(PROMPT, return_tensors="pt")

    # ------------------------------------------------------------
    # 5.b — Greedy
    # ------------------------------------------------------------
    print("=" * 80)
    print("5.b — Greedy decoding (3 runs)")
    print("=" * 80)

    for i in range(1, 4):
        # greedy = do_sample=False (par défaut) + num_beams=1
        out = model.generate(
            **inputs,
            max_length=MAX_LEN,
            do_sample=False,
        )
        print(f"[Run {i}]")
        print(decode(tokenizer, out))
        print("-" * 40)

    # ------------------------------------------------------------
    # 5.c — Sampling (temp=0.7, top_k=50, top_p=0.95) sur 5 seeds
    # ------------------------------------------------------------
    print()
    print("=" * 80)
    print("5.c — Sampling (temp=0.7, top_k=50, top_p=0.95) — 5 seeds")
    print("=" * 80)

    def generate_sampling(seed, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=None):
        set_seed(seed)
        kwargs = dict(
            **inputs,
            max_length=MAX_LEN,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = repetition_penalty

        out = model.generate(**kwargs)
        return decode(tokenizer, out)

    for s in [1, 2, 3, 4, 5]:
        print("SEED", s)
        print(generate_sampling(s, temperature=0.7, top_k=50, top_p=0.95))
        print("-" * 40)

    # ------------------------------------------------------------
    # 5.d — Repetition penalty
    # ------------------------------------------------------------
    print()
    print("=" * 80)
    print("5.d — Repetition penalty comparison (same seed)")
    print("=" * 80)

    seed_test = 7
    print("Seed:", seed_test)
    txt_no_pen = generate_sampling(seed_test, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=None)
    txt_pen = generate_sampling(seed_test, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=2.0)

    print("[No repetition_penalty]")
    print(txt_no_pen)
    print("-" * 40)
    print("[With repetition_penalty=2.0]")
    print(txt_pen)

    # ------------------------------------------------------------
    # 5.e — Temp très basse vs très haute
    # ------------------------------------------------------------
    print()
    print("=" * 80)
    print("5.e — Temperature extremes (same seed)")
    print("=" * 80)

    seed_temp = 9
    print("Seed:", seed_temp)
    txt_t_low = generate_sampling(seed_temp, temperature=0.1, top_k=50, top_p=0.95)
    txt_t_high = generate_sampling(seed_temp, temperature=2.0, top_k=50, top_p=0.95)

    print("[temperature=0.1]")
    print(txt_t_low)
    print("-" * 40)
    print("[temperature=2.0]")
    print(txt_t_high)

    # ------------------------------------------------------------
    # 5.f — Beam search
    # ------------------------------------------------------------
    print()
    print("=" * 80)
    print("5.f — Beam search (num_beams=5)")
    print("=" * 80)

    set_seed(SEED)
    out_beam = model.generate(
        **inputs,
        max_length=MAX_LEN,
        num_beams=5,
        early_stopping=True,
        do_sample=False,
    )
    print(decode(tokenizer, out_beam))

    # ------------------------------------------------------------
    # 5.g — Beam search timing: 5 vs 10 vs 20
    # ------------------------------------------------------------
    print()
    print("=" * 80)
    print("5.g — Beam search timing (approx)")
    print("=" * 80)

    for beams in [5, 10, 20]:
        set_seed(SEED)
        t0 = time.perf_counter()
        _ = model.generate(
            **inputs,
            max_length=MAX_LEN,
            num_beams=beams,
            early_stopping=True,
            do_sample=False,
        )
        t1 = time.perf_counter()
        print(f"num_beams={beams} -> {t1 - t0:.3f}s")


if __name__ == "__main__":
    main()
