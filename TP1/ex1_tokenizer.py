from transformers import GPT2Tokenizer

def show_tokens_and_ids(tokenizer, text, title=None):
    if title:
        print("=" * 80)
        print(title)
        print("=" * 80)

    print("Texte:", repr(text))
    print()

    # 2.a — tokens
    tokens = tokenizer.tokenize(text)
    print("Tokens:")
    print(tokens)
    print()

    # 2.b — IDs + vérification décodage token-par-token
    token_ids = tokenizer.encode(text)
    print("Token IDs:", token_ids)
    print()

    print("Détails par token (id -> decode([id])):")
    for tid in token_ids:
        decoded = tokenizer.decode([tid])
        print(f"{tid}\t{repr(decoded)}")
    print()

    # Vérif : reconstruction complète
    reconstructed = tokenizer.decode(token_ids)
    print("Reconstruction (decode(ids)):", repr(reconstructed))
    print()

    return tokens, token_ids


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 2.a / 2.b / 2.c
    phrase1 = "Artificial intelligence is metamorphosing the world!"
    tokens1, ids1 = show_tokens_and_ids(tokenizer, phrase1, title="Phrase 1")

    # 2.d
    phrase2 = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."
    tokens2, ids2 = show_tokens_and_ids(tokenizer, phrase2, title="Phrase 2")

    # Extraction des sous-tokens du mot long (2.d)
    # Méthode robuste: tokeniser uniquement le mot long
    long_word = "antidisestablishmentarianism"
    long_tokens = tokenizer.tokenize(long_word)
    long_ids = tokenizer.encode(long_word, add_special_tokens=False)

    print("=" * 80)
    print("Analyse du mot long")
    print("=" * 80)
    print("Mot:", long_word)
    print("Sous-tokens:", long_tokens)
    print("Nombre de sous-tokens:", len(long_tokens))
    print("IDs:", long_ids)
    print("Reconstruction:", repr(tokenizer.decode(long_ids)))


if __name__ == "__main__":
    main()
