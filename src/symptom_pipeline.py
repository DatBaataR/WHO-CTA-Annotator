"""
Symptom Extraction Pipeline for WHO Epidemiological Datasets.
Combines:
  1. Coded-data bypass heuristic (semicolon, pipe, underscore, and short comma-separated codes)
  2. Multilingual translation via M2M100-418M (automatic language detection + translation to English)
  3. Medical NER extraction via HUMADEX/english_medical_ner with custom BIOES full-word span merger
  4. SpaCy lemmatization, severity/filler-adverb filtering
  5. Fuzzy clustering (>85 Levenshtein similarity) to consolidate spelling variations into boolean columns
"""

import math
import os
import re
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import langdetect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    from thefuzz import fuzz
    HAS_THEFUZZ = True
except ImportError:
    import difflib
    HAS_THEFUZZ = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    nlp = None
    HAS_SPACY = False

_NER_STATE = {"loaded": False, "raw_pipeline": None, "simple_pipeline": None}
_TRANSLATOR_STATE = {"loaded": False, "tokenizer": None, "model": None}

NULL_VALUES = {
    "none", "nan", "null", "nothing", "n/a", "na", "nil", "unknown", "other",
    "public information are not available", "public information", "not available",
    "no", "false", "0", "1"
}

SEVERITY_WORDS = {
    "profuse", "intermittant", "intermittent", "modest", "severe", "mild", "acute", "chronic"
}

FILLER_ADVERBS = {
    "very", "quite", "too", "so", "more", "most", "less", "extremely", "highly", "slightly", "really"
}


def _fuzzy_ratio(s1, s2):
    if HAS_THEFUZZ:
        return fuzz.ratio(s1, s2)
    return int(round(difflib.SequenceMatcher(None, s1, s2).ratio() * 100))


def _ensure_ner_loaded(base_dir=None):
    if _NER_STATE["loaded"]:
        return _NER_STATE["raw_pipeline"]
    _NER_STATE["loaded"] = True
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        candidates = []
        if base_dir:
            candidates.append(os.path.join(base_dir, "models", "english_medical_ner"))
            candidates.append(os.path.join(base_dir, "english_medical_ner"))
        # Fallback to sibling thesis repo if available on this machine, then HuggingFace Hub
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "english_medical_ner")))
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RECA-paper", "Semtab", "experiment", "english_medical_ner")))
        candidates.append("HUMADEX/english_medical_ner")

        for model_path in candidates:
            if model_path == "HUMADEX/english_medical_ner" or os.path.isdir(model_path):
                try:
                    tok = AutoTokenizer.from_pretrained(model_path)
                    mod = AutoModelForTokenClassification.from_pretrained(model_path)
                    _NER_STATE["raw_pipeline"] = pipeline("ner", model=mod, tokenizer=tok, aggregation_strategy="none")
                    _NER_STATE["simple_pipeline"] = pipeline("ner", model=mod, tokenizer=tok, aggregation_strategy="simple")
                    print(f"[SymptomPipeline] Loaded Medical NER model from: {model_path}")
                    break
                except Exception:
                    continue
    except Exception as e:
        print(f"[SymptomPipeline] Warning: Medical NER pipeline unavailable ({e}). Using rule/spacy fallback.")
    return _NER_STATE["raw_pipeline"]


def _ensure_translator_loaded(base_dir=None):
    if _TRANSLATOR_STATE["loaded"]:
        return _TRANSLATOR_STATE["model"] is not None
    _TRANSLATOR_STATE["loaded"] = True
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        candidates = []
        if base_dir:
            candidates.append(os.path.join(base_dir, "models", "m2m100_418M"))
            candidates.append(os.path.join(base_dir, "m2m100_418M"))
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "m2m100_418M")))
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RECA-paper", "Semtab", "experiment", "m2m100_418M")))
        candidates.append("facebook/m2m100_418M")

        for model_path in candidates:
            if model_path == "facebook/m2m100_418M" or os.path.isdir(model_path):
                try:
                    # Only auto-load local path unless explicitly downloaded via setup_models.py to avoid unexpected 1.6GB wait
                    if model_path == "facebook/m2m100_418M" and not os.environ.get("AUTO_DOWNLOAD_M2M100"):
                        continue
                    tok = AutoTokenizer.from_pretrained(model_path)
                    mod = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                    _TRANSLATOR_STATE["tokenizer"] = tok
                    _TRANSLATOR_STATE["model"] = mod
                    print(f"[SymptomPipeline] Loaded M2M100 Translator from: {model_path}")
                    break
                except Exception:
                    continue
    except Exception:
        pass
    return _TRANSLATOR_STATE["model"] is not None


def _is_valid_symptom_token(word):
    if not word:
        return False
    w_clean = word.strip(" .,;:!?-|_\"'\n\t").lower()
    if len(w_clean) <= 2 or "#" in w_clean:
        return False
    if w_clean in NULL_VALUES:
        return False
    if not any(ch.isalpha() for ch in w_clean):
        return False
    return True


def _is_filler_adv(token):
    if token.pos_ != "ADV":
        return False
    lem = token.lemma_.lower()
    return lem in FILLER_ADVERBS or lem.endswith("ly")


def _clean_with_spacy(raw_word):
    raw_word = raw_word.lower().strip(" .,;:!?-|_\"'\n\t")
    if not _is_valid_symptom_token(raw_word):
        return ""

    if HAS_SPACY and nlp is not None:
        doc = nlp(raw_word)
        cleaned = [
            token.lemma_
            for token in doc
            if token.pos_ not in ["ADJ", "DET", "PUNCT"]
            and not _is_filler_adv(token)
            and not token.is_punct
            and token.lemma_.lower() not in SEVERITY_WORDS
        ]
        while cleaned and cleaned[0].lower() in {"and", "or", "of", "in", "to", "with"}:
            cleaned.pop(0)
        while cleaned and cleaned[-1].lower() in {"and", "or", "of", "in", "to", "with"}:
            cleaned.pop()

        if not cleaned:
            cleaned = [
                token.lemma_
                for token in doc
                if not token.is_punct
                and token.pos_ not in ["PUNCT", "DET"]
                and token.lemma_.lower() not in SEVERITY_WORDS
            ]

        if len(cleaned) == 1 and cleaned[0].endswith("ing") and len(doc) > 1:
            cleaned = [nlp(cleaned[0])[0].lemma_]

        final_word = " ".join(cleaned).strip(" .,;:!?-|_\"'\n\t")
    else:
        tokens = [t for t in raw_word.split() if t not in SEVERITY_WORDS and t not in FILLER_ADVERBS]
        final_word = " ".join(tokens).strip(" .,;:!?-|_\"'\n\t")

    final_word = re.sub(r"\s+", " ", final_word).strip()
    return final_word if _is_valid_symptom_token(final_word) else ""


def _extract_bioes_problem_spans_single(text, base_dir=None):
    raw_ner = _ensure_ner_loaded(base_dir)
    if raw_ner is None:
        # Fallback if NER model is not loaded: split by comma/conjunctions
        parts = re.split(r"[,;]|(?:\band\b)|(?:\bwith\b)", text)
        return [p.strip() for p in parts if p.strip()]

    tokens = raw_ner(text)
    words = []
    for tok in tokens:
        ent = tok["entity"]
        tag, label = (ent.split("-", 1) if "-" in ent else ("O", "O"))
        if label != "PROBLEM":
            continue
        s, e = tok["start"], tok["end"]
        if text[s:e].startswith("#"):
            continue
        while s > 0 and text[s - 1].isalnum():
            s -= 1
        while e < len(text) and text[e].isalnum():
            e += 1
        if words and s <= words[-1]["end"]:
            words[-1]["end"] = max(words[-1]["end"], e)
            if words[-1]["tag"] in ("B", "I") and tag == "E":
                words[-1]["tag"] = "E"
        else:
            words.append({"start": s, "end": e, "tag": tag})

    spans = []
    for w in words:
        between = text[spans[-1]["end"]:w["start"]] if spans else None
        if spans and between is not None and between.strip() == "" and (
            w["tag"] in ("I", "E") or spans[-1]["last_tag"] in ("B", "I")
        ):
            spans[-1]["end"] = max(spans[-1]["end"], w["end"])
            spans[-1]["last_tag"] = w["tag"]
        else:
            spans.append({"start": w["start"], "end": w["end"], "last_tag": w["tag"]})
    return [text[s["start"]:s["end"]] for s in spans]


def _extract_bioes_problem_spans(text, base_dir=None):
    if len(text) <= 400:
        return _extract_bioes_problem_spans_single(text, base_dir=base_dir)

    spans = []
    parts = re.split(r"(?<=[.,;])\s+|\s+", text)
    current = []
    curr_len = 0
    for p in parts:
        if curr_len + len(p) + 1 > 350 and current:
            spans.extend(_extract_bioes_problem_spans_single(" ".join(current), base_dir=base_dir))
            current = [p]
            curr_len = len(p)
        else:
            current.append(p)
            curr_len += len(p) + 1
    if current:
        spans.extend(_extract_bioes_problem_spans_single(" ".join(current), base_dir=base_dir))
    return spans


def extract_symptoms_from_text(text, lang="en", base_dir=None):
    """Extracts normalized symptom names from a single cell value."""
    if text is None or (isinstance(text, float) and math.isnan(text)) or pd.isna(text):
        return []

    original_text = str(text).strip()
    if not original_text:
        return []

    original_text = re.sub(r'^"?[A-Za-z0-9_ ]+\s+:\s+', '', original_text).strip('"').strip()
    whole_clean = original_text.strip(" .,;:!?-|_\"'\n\t").lower()
    if not whole_clean or whole_clean in NULL_VALUES or not any(ch.isalpha() for ch in whole_clean):
        return []

    # 1. Coded Data Bypass Heuristic
    is_coded_data = False
    chunks = []

    if ";" in original_text or "|" in original_text:
        is_coded_data = True
        raw_chunks = re.split(r"[;|,]+", original_text)
        for rc in raw_chunks:
            rc_strip = rc.strip()
            if not rc_strip:
                continue
            if rc_strip.isupper() and len(rc_strip.split()) > 1:
                chunks.extend(rc_strip.split())
            else:
                chunks.append(rc_strip)
    elif "_" in original_text:
        u_parts = [p.strip() for p in original_text.split("_") if p.strip()]
        if u_parts and all(len(p.split()) == 1 for p in u_parts):
            is_coded_data = True
            chunks = u_parts
    elif "," in original_text:
        c_parts = [c.strip() for c in original_text.split(",") if c.strip()]
        if c_parts:
            avg_words = sum(len(c.split()) for c in c_parts) / len(c_parts)
            max_words = max(len(c.split()) for c in c_parts)
            if avg_words < 3 and max_words <= 3:
                is_coded_data = True
                chunks = c_parts
    else:
        words = original_text.split()
        if len(words) <= 3:
            is_coded_data = True
            chunks = [original_text]
        elif original_text.isupper() and len(set(words)) <= max(3, len(words) // 2):
            is_coded_data = True
            chunks = words
        else:
            non_flag_parts = [
                p.strip()
                for p in re.split(r"\b[01]\b|(?=\b[A-Z][a-z]+)", original_text)
                if p.strip()
            ]
            if len(non_flag_parts) > 1 and len(set(non_flag_parts)) == 1 and len(non_flag_parts[0].split()) <= 3:
                is_coded_data = True
                chunks = non_flag_parts

    if is_coded_data:
        symptoms = []
        for c in chunks:
            final_word = _clean_with_spacy(c)
            if final_word:
                symptoms.append(final_word)
        return symptoms

    # 2. Natural Language Pipeline
    inline_coded = []
    for match in re.findall(r"\b[A-Z]{3,}(?:_[A-Z]{3,})+\b", original_text):
        for code_part in match.split("_"):
            cleaned_code = _clean_with_spacy(code_part)
            if cleaned_code:
                inline_coded.append(cleaned_code)

    text_norm = original_text.replace(";", ", ").replace("|", ", ").replace("_", ", ").lower()

    if lang != "en" and _ensure_translator_loaded(base_dir):
        try:
            tok = _TRANSLATOR_STATE["tokenizer"]
            mod = _TRANSLATOR_STATE["model"]
            tok.src_lang = lang
            encoded = tok(text_norm, return_tensors="pt")
            generated_tokens = mod.generate(
                **encoded,
                forced_bos_token_id=tok.get_lang_id("en")
            )
            target_text = tok.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        except Exception:
            target_text = text_norm
    else:
        target_text = text_norm

    raw_spans = _extract_bioes_problem_spans(target_text, base_dir=base_dir)
    symptoms = list(inline_coded)
    for span in raw_spans:
        final_word = _clean_with_spacy(span)
        if final_word:
            symptoms.append(final_word)

    return symptoms


def preview_symptoms_expansion(df, target_col="Symptoms", base_dir=None):
    """
    Analyzes a column in `df` and returns the dictionary of proposed symptom boolean columns
    and summary statistics without mutating `df`.
    Returns:
        dict with keys:
          - "target_col": str
          - "detected_language": str
          - "columns": dict mapping "symptom: <name>" -> list of bools (length = len(df))
          - "counts": dict mapping "symptom: <name>" -> int (number of True rows)
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")

    known_symptoms = []
    new_cols_data = {}
    extraction_cache = {}

    sample_texts = df[target_col].dropna().astype(str).head(50).tolist()
    natural_samples = [s for s in sample_texts if not s.isupper()]
    combined_sample = " ".join(natural_samples).replace(";", " ").replace("_", " ").lower()
    column_lang = "en"
    if combined_sample.strip() and HAS_LANGDETECT:
        try:
            column_lang = langdetect.detect(combined_sample)
        except Exception:
            column_lang = "en"

    for pos_idx, (_, row_val) in enumerate(df[target_col].items()):
        if pd.isna(row_val) or str(row_val).strip() == "":
            continue

        text = str(row_val)
        if text not in extraction_cache:
            extraction_cache[text] = extract_symptoms_from_text(text, lang=column_lang, base_dir=base_dir)
        extracted_symptoms = extraction_cache[text]

        for raw_sym in extracted_symptoms:
            matched_sym = None
            for known in known_symptoms:
                if _fuzzy_ratio(raw_sym, known) > 85:
                    matched_sym = known
                    break

            if not matched_sym:
                matched_sym = raw_sym
                known_symptoms.append(matched_sym)
                new_cols_data[f"symptom: {matched_sym}"] = [False] * len(df)

            new_cols_data[f"symptom: {matched_sym}"][pos_idx] = True

    counts = {col_name: int(sum(bool_list)) for col_name, bool_list in new_cols_data.items()}
    return {
        "target_col": target_col,
        "detected_language": column_lang,
        "columns": new_cols_data,
        "counts": counts,
    }


def expand_symptoms_dataframe(df, target_col="Symptoms", selected_symptom_cols=None, custom_symptom_cols=None, drop_original=True, base_dir=None):
    """
    Expands `target_col` into boolean `symptom: <name>` columns.
    If `selected_symptom_cols` is provided, only keeps those generated columns.
    If `custom_symptom_cols` is provided (list of symptom names or 'symptom: <name>'),
    adds those columns by matching rows whose cell text contains the symptom keyword/fuzzy match.
    """
    preview = preview_symptoms_expansion(df, target_col=target_col, base_dir=base_dir)
    new_cols_data = preview["columns"]

    out_df = df.copy()

    # Filter to user-selected generated columns if specified
    if selected_symptom_cols is not None:
        selected_set = set(selected_symptom_cols)
        new_cols_data = {k: v for k, v in new_cols_data.items() if k in selected_set}

    # Add any custom user-defined symptom columns
    if custom_symptom_cols:
        for custom_raw in custom_symptom_cols:
            custom_clean = str(custom_raw).strip()
            if not custom_clean:
                continue
            if not custom_clean.lower().startswith("symptom:"):
                col_title = f"symptom: {custom_clean.lower()}"
                keyword = custom_clean.lower()
            else:
                col_title = f"symptom: {custom_clean.split(':', 1)[1].strip().lower()}"
                keyword = custom_clean.split(":", 1)[1].strip().lower()

            if col_title not in new_cols_data:
                bool_vals = []
                for _, row_val in out_df[target_col].items():
                    if pd.isna(row_val) or not str(row_val).strip():
                        bool_vals.append(False)
                    else:
                        cell_lower = str(row_val).lower()
                        matched = (keyword in cell_lower)
                        if not matched:
                            tokens = re.split(r"[\s,;|_]+", cell_lower)
                            matched = any(_fuzzy_ratio(keyword, tok) > 85 for tok in tokens if tok)
                        bool_vals.append(bool(matched))
                new_cols_data[col_title] = bool_vals

    for col_name, bool_list in new_cols_data.items():
        out_df[col_name] = bool_list

    if drop_original and target_col in out_df.columns:
        out_df.drop(columns=[target_col], inplace=True)

    return out_df
