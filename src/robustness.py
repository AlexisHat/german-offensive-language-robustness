"""Relate perturbation-induced performance degradation to tokenizer/vocabulary behavior."""


def token_fragmentation_ratio(tokenizer, clean_texts, perturbed_texts):
    """Ratio of WordPiece token counts (perturbed / clean), excluding special tokens.

    A ratio above 1 means the perturbed text tokenizes into more subword
    pieces than the original, i.e. words got fragmented.
    """
    clean_lengths = [
        len(ids) for ids in tokenizer(list(clean_texts), add_special_tokens=False)["input_ids"]
    ]
    perturbed_lengths = [
        len(ids) for ids in tokenizer(list(perturbed_texts), add_special_tokens=False)["input_ids"]
    ]
    return [p / c for c, p in zip(clean_lengths, perturbed_lengths) if c > 0]


def oov_rate(vectorizer, texts):
    """Share of a text's unigram tokens that are absent from the vectorizer's vocabulary.

    Uses the vectorizer's own analyzer, so preprocessing (lowercasing,
    tokenization) matches exactly what was used at training time. Bigram
    tokens (containing a space) are excluded, since "words missing from the
    vocabulary" is a word-level, not phrase-level, notion.
    """
    vocabulary = vectorizer.vocabulary_
    analyzer = vectorizer.build_analyzer()

    rates = []
    for text in texts:
        unigrams = [token for token in analyzer(text) if " " not in token]
        if not unigrams:
            rates.append(0.0)
            continue
        n_oov = sum(1 for token in unigrams if token not in vocabulary)
        rates.append(n_oov / len(unigrams))
    return rates
