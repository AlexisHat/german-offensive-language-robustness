"""Controlled, label-preserving text perturbations for robustness evaluation."""

import math
import random
import re

# clean_text() normalizes every mention to exactly this literal token
PROTECTED_TOKEN_PATTERN = re.compile(r"@USER")

VOWELS = set("aeiouäöüAEIOUÄÖÜ")

UMLAUT_MAP = {
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
}

# Approximate German QWERTZ keyboard adjacency, used to pick realistic
# substitution/insertion typos
KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "rzfg",
    "z": "tugh", "u": "zihj", "i": "uojk", "o": "ipkl", "p": "oü",
    "ü": "p", "a": "qwsy", "s": "weadxy", "d": "ersfxc", "f": "rtdgcv",
    "g": "tzfhvb", "h": "zugjbn", "j": "uihknm", "k": "iojlm",
    "l": "opkö", "ö": "läp", "ä": "ö", "y": "asx", "x": "sdyc",
    "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn",
    "ß": "s",
}

# Informal German register substitutions. Only unambiguous,
# non-dialect-coded, register-only swaps: none of them add or remove
# profanity/intensity, since that could change how offensive a tweet
# reads and break label preservation.
INFORMAL_SUBSTITUTIONS = {
    "nicht": "nich",
    "nichts": "nix",
    "ist": "is",
    "sind": "sin",
    "habe": "hab",
    "haben": "ham",
    "hatte": "hatt",
    "eine": "ne",
    "einen": "nen",
    "einem": "nem",
    "einer": "ner",
    "etwas": "was",
    "gerade": "grad",
    "jetzt": "jetz",
    "mal": "ma",
    "vielleicht": "vllt",
    "ungefähr": "circa",
    "eigentlich": "eigtl",
    "wahrscheinlich": "wahrsch",
    "beziehungsweise": "bzw",
    "zum Beispiel": "z. B.",
    "keine Ahnung": "kA",
    "auf jeden Fall": "auf jeden",
    "okay": "ok",
    "ja": "jo",
    "nein": "nö",
    "Hallo": "Hi",
    "Tschüss": "ciao",
    "Entschuldigung": "sorry",
    "Danke": "thx",
}


def find_protected_spans(text):
    """Return the (start, end) character spans of "@USER" tokens in text."""
    return [m.span() for m in PROTECTED_TOKEN_PATTERN.finditer(text)]


def make_seed(row_index, intensity, base_seed):
    """Create a reproducible seed for one row and intensity level."""
    return base_seed + row_index * 1000 + round(intensity * 100)


def _is_protected(index, spans):
    return any(start <= index < end for start, end in spans)


def apply_typos(text, intensity, seed):
    """Add random character-level typos outside protected spans."""
    rng = random.Random(seed)
    protected = find_protected_spans(text)
    eligible = [i for i, ch in enumerate(text) if ch.isalpha() and not _is_protected(i, protected)]

    n_to_perturb = math.ceil(intensity * len(eligible))
    if n_to_perturb == 0:
        return text

    positions = rng.sample(eligible, n_to_perturb)
    chars = list(text)

    # Process right-to-left: insertions/deletions change the string length,
    # so earlier positions must be handled last to stay valid.
    for i in sorted(positions, reverse=True):
        op = rng.choice(["substitute", "delete", "insert", "transpose"])
        ch = chars[i]
        lower = ch.lower()
        neighbors = KEYBOARD_NEIGHBORS.get(lower, lower)

        if op == "substitute":
            replacement = rng.choice(neighbors)
            chars[i] = replacement.upper() if ch.isupper() else replacement
        elif op == "delete":
            del chars[i]
        elif op == "insert":
            insertion = rng.choice(neighbors)
            chars.insert(i, insertion.upper() if ch.isupper() else insertion)
        elif op == "transpose" and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

    return "".join(chars)


def apply_casing_noise(text, intensity, seed):
    """Randomly change the case of letters outside protected spans."""
    rng = random.Random(seed)
    protected = find_protected_spans(text)
    eligible = [i for i, ch in enumerate(text) if ch.isalpha() and not _is_protected(i, protected)]

    n_to_perturb = math.ceil(intensity * len(eligible))
    if n_to_perturb == 0:
        return text

    chars = list(text)
    for i in rng.sample(eligible, n_to_perturb):
        chars[i] = chars[i].swapcase()
    return "".join(chars)


def apply_umlaut_variants(text, intensity, seed):
    """Replace selected umlauts and ß with ASCII spellings."""
    rng = random.Random(seed)
    protected = find_protected_spans(text)
    eligible = [i for i, ch in enumerate(text) if ch in UMLAUT_MAP and not _is_protected(i, protected)]

    n_to_perturb = math.ceil(intensity * len(eligible))
    if n_to_perturb == 0:
        return text

    chars = list(text)
    for i in rng.sample(eligible, n_to_perturb):
        chars[i] = UMLAUT_MAP[chars[i]]
    return "".join(chars)


def apply_elongation(text, intensity, seed):
    """Stretch letters in selected words, for example nein to neiiin."""
    rng = random.Random(seed)
    protected = find_protected_spans(text)

    tokens = list(re.finditer(r"\S+", text))
    eligible_idx = [
        idx for idx, m in enumerate(tokens)
        if len(m.group()) >= 2 and not _is_protected(m.start(), protected)
    ]

    n_to_perturb = math.ceil(intensity * len(eligible_idx))
    if n_to_perturb == 0:
        return text

    chosen = rng.sample(eligible_idx, n_to_perturb)
    chars = list(text)

    # Process right-to-left: each elongation lengthens the string, so
    # earlier token positions must be handled last to stay valid.
    for idx in sorted(chosen, reverse=True):
        m = tokens[idx]
        word = m.group()

        target_pos = next(
            (j for j in range(len(word) - 1, -1, -1) if word[j] in VOWELS),
            len(word) - 1,
        )
        repeat_count = rng.choice([2, 3])
        abs_pos = m.start() + target_pos
        chars[abs_pos:abs_pos + 1] = [chars[abs_pos]] * (repeat_count + 1)

    return "".join(chars)


def apply_slang_substitution(text, intensity, seed, lexicon=INFORMAL_SUBSTITUTIONS):
    """Replace selected words or phrases with informal alternatives."""
    rng = random.Random(seed)
    protected = find_protected_spans(text)

    matches = []
    for key, value in lexicon.items():
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            if not _is_protected(m.start(), protected):
                matches.append((m.start(), m.end(), value))

    n_to_perturb = math.ceil(intensity * len(matches))
    if n_to_perturb == 0:
        return text

    chosen = rng.sample(matches, n_to_perturb)
    # Process right-to-left so earlier match positions stay valid after
    # replacements of a different length.
    for start, end, value in sorted(chosen, key=lambda m: m[0], reverse=True):
        text = text[:start] + value + text[end:]

    return text


PERTURBATION_FUNCTIONS = {
    "typo": apply_typos,
    "casing": apply_casing_noise,
    "umlaut": apply_umlaut_variants,
    "elongation": apply_elongation,
    "slang": apply_slang_substitution,
}


def perturb(text, kind, intensity, seed):
    """Apply one perturbation type to a text."""
    if kind not in PERTURBATION_FUNCTIONS:
        raise ValueError(f"Unknown perturbation kind: {kind!r}")
    return PERTURBATION_FUNCTIONS[kind](text, intensity, seed)