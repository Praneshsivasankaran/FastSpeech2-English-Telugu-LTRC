"""Reproducible English/Telugu IPA front end, preserving source text separately.

Uses eSpeak NG through BFA's ph66 mapper; IPA labels remain the original
pronunciations. Compound acoustic classes are merged back into their IPA phone.
Unknown/dropped phonemes raise rather than silently becoming padding.
"""
from functools import lru_cache
import re
import unicodedata


def configure_espeak():
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())


def language_code(language):
    try:
        return {"English": "en-us", "english": "en-us", "en": "en-us", "en-us": "en-us",
                "Telugu": "te", "telugu": "te", "te": "te"}[language]
    except KeyError:
        raise ValueError(f"Unsupported language: {language!r}")


def normalize_text(text, language):
    """Normalize whitespace and time meridiems without altering source records.

    eSpeak reads numeric dates/times in the selected language. English AM/PM
    following a digit is expanded to letter names (A M / P M), preventing the
    word "am" pronunciation. Deliberately split words are never repaired.
    """
    code = language_code(language)
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
    if code == "en-us":
        normalized = re.sub(r"(?<=\d)\s*([AP])\.?M\.?\b",
                            lambda m: " " + m.group(1).upper() + " M", normalized,
                            flags=re.IGNORECASE)
    return normalized


def configure_phonemizer(p):
    """Fix upstream alpha breaker and phone joins across eSpeak number expansions."""
    from phonemizer.separator import Separator
    p.unicode_enable = True
    p.remove_noise_phonemes = False
    if not getattr(p.backend, "_word_join_fixed", False):
        original = p.backend.phonemize
        def phonemize_with_word_boundaries(text, *args, **kwargs):
            kwargs["separator"] = Separator(phone="|", word=" ", syllable="")
            return [re.sub(r"\s+", "|", line) for line in original(text, *args, **kwargs)]
        p.backend.phonemize = phonemize_with_word_boundaries
        p.backend._word_join_fixed = True
    # BFA's generic map omits Telugu phones: retain the original IPA label
    # while assigning a nearest acoustic class. These are alignment-only
    # approximations; the TTS vocabulary still distinguishes the IPA phones.
    if p.language == "te" and not getattr(p, "_telugu_inventory_fixed", False):
        from types import FunctionType, MethodType
        original_method = p.phonemize_sentence.__func__
        scope = dict(original_method.__globals__)
        overrides = {"ʲ": "j", "ɡʰ": "g", "ɟʰ": "ʒ"}
        scope["phoneme_mapper"] = dict(scope["phoneme_mapper"], **overrides)
        base_mapper = scope["get_compound_phoneme_mapping"]
        def telugu_acoustic_mapping(phone):
            return [overrides[phone]] if phone in overrides else base_mapper(phone)
        scope["get_compound_phoneme_mapping"] = telugu_acoustic_mapping
        # Scope only this instance's unchanged BFA algorithm; never modify
        # shared package dictionaries or the English frontend.
        scoped_method = FunctionType(original_method.__code__, scope,
                                     original_method.__name__, original_method.__defaults__,
                                     original_method.__closure__)
        p.phonemize_sentence = MethodType(scoped_method, p)
        p._telugu_inventory_fixed = True
    return p


@lru_cache(maxsize=2)
def _phonemizer(language):
    configure_espeak()
    from bournemouth_aligner.ipamappers.ph66_phonemeizer import Phonemizer
    return configure_phonemizer(Phonemizer(language=language_code(language), remove_noise_phonemes=False, unicode_enable=True, verbose=0))


def alignment_text(text, language):
    """Offer optional silence at each word boundary to the acoustic decoder."""
    return " <SIL> ".join(normalize_text(text, language).split())


def phoneme_detail(text, language):
    p = _phonemizer(language_code(language))
    detail = p.phonemize_sentence(alignment_text(text, language))
    # ph66 noise index is distinct from SIL. A mapper miss is not a valid phone.
    if p.phoneme_index["noise"] in detail["ph66"]:
        raise ValueError(f"eSpeak produced an unmapped phone for {language}: {text!r}")
    if not detail["eipa"] or all(x == "SIL" for x in detail["eipa"]):
        raise ValueError("No speech phonemes produced")
    return detail


def fold_phone_groups(detail):
    """Return (IPA label, raw target indices); join compound-class continuations."""
    groups = []
    for i, phone in enumerate(detail["eipa"]):
        if phone == "-":
            if not groups or groups[-1][0] == "SIL":
                raise ValueError("Orphan acoustic compound continuation")
            groups[-1][1].append(i)
        elif phone == "SIL" and groups and groups[-1][0] == "SIL":
            groups[-1][1].append(i)
        else:
            groups.append((phone, [i]))
    if groups[0][0] != "SIL":
        groups.insert(0, ("SIL", []))
    if groups[-1][0] != "SIL":
        groups.append(("SIL", []))
    return groups


def phonemize(text, language):
    return [p for p, _ in fold_phone_groups(phoneme_detail(text, language))]
