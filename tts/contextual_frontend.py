"""Experimental sentence-context frontend, separate from the training frontend.

This opt-in candidate keeps BFA's phone inventory, mapping and SIL slots. It
uses eSpeak's full-sentence context only when the resulting word count can be
mapped back unambiguously. Otherwise it records a whole-utterance fallback to
the existing word-level frontend. It does not repair spelling or split words.
Stress markers remain disabled by the existing BFA backend; contextual vowel
selection is not a guarantee of correct acoustic stress or pronunciation.
"""
from __future__ import annotations

import re
import unicodedata

from . import frontend as training_frontend


def phoneme_detail(text, language):
    """Return BFA-compatible detail plus explicit ``contextual_frontend`` metadata.

    Every call owns a fresh phonemizer instance. No cached training instance,
    installed package, shared mapper dictionary or training source is modified.
    """
    from bournemouth_aligner.ipamappers.ph66_phonemeizer import Phonemizer
    from phonemizer.separator import Separator

    training_frontend.configure_espeak()
    code = training_frontend.language_code(language)
    phonemizer = Phonemizer(language=code, remove_noise_phonemes=False,
                           unicode_enable=True, verbose=0)
    # Capture the unwrapped backend before installing BFA's word-boundary fix.
    raw_backend = phonemizer.backend.phonemize
    phonemizer = training_frontend.configure_phonemizer(phonemizer)
    word_level_backend = phonemizer.backend.phonemize
    metadata = {
        "candidate": "sentence-context-v1",
        "language": code,
        "status": "not_called",
        "fallback_reason": None,
        "stress_markers_encoded": False,
        "training_frontend_unchanged": True,
        "mapping": "Existing BFA mapping and original SIL word slots",
    }

    def context_backend(words, *args, **kwargs):
        if not isinstance(words, list) or not all(isinstance(word, str) for word in words):
            raise TypeError("Contextual frontend requires BFA's list of words")
        non_silence = [word for word in words if word != "<SIL>"]
        sentence = " ".join(non_silence)
        metadata.update(input_non_silence_words=non_silence,
                        input_word_count=len(non_silence), contextual_input=sentence)
        context = raw_backend(
            [sentence], separator=Separator(phone="|", word=" ", syllable=""),
            strip=True, njobs=1,
        )
        if len(context) != 1:
            raise ValueError("Unexpected sentence-context backend result count")
        chunks = re.split(r"\s+", context[0].strip()) if context[0].strip() else []
        chunks = [unicodedata.normalize("NFC", chunk) for chunk in chunks]
        metadata.update(contextual_word_count=len(chunks), contextual_word_phone_strings=chunks)
        if len(chunks) != len(non_silence) or any(not chunk.replace("|", "") for chunk in chunks):
            metadata.update(status="fallback_word_level",
                            fallback_reason="contextual_word_count_mismatch",
                            contextual_word_count_matches=False)
            return word_level_backend(words, *args, **kwargs)
        metadata.update(status="contextual", contextual_word_count_matches=True)
        iterator = iter(chunks)
        # BFA itself handles the SIL slots, so their backend output is unused.
        return ["" if word == "<SIL>" else next(iterator) for word in words]

    phonemizer.backend.phonemize = context_backend
    detail = phonemizer.phonemize_sentence(training_frontend.alignment_text(text, language))
    if phonemizer.phoneme_index["noise"] in detail["ph66"]:
        raise ValueError(f"Contextual frontend produced an unmapped phone for {language}: {text!r}")
    if not detail["eipa"] or all(phone == "SIL" for phone in detail["eipa"]):
        raise ValueError("No speech phonemes produced")
    detail["contextual_frontend"] = metadata
    return detail


def phonemize_with_metadata(text, language):
    detail = phoneme_detail(text, language)
    return {"phones": [phone for phone, _ in training_frontend.fold_phone_groups(detail)],
            "metadata": detail["contextual_frontend"]}


def phonemize(text, language):
    """Return only phones for an explicitly selected experimental inference path."""
    return phonemize_with_metadata(text, language)["phones"]
