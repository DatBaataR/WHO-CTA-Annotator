"""
Hybrid Routing Engine for WHO Epidemiological Column Type Annotation.
Supports:
  1. RECA + Hybrid Routing (ArcheType-Gemini LLM escalation on disagreement / low confidence / weak classes + table-level soft penalty)
  2. RF + Hybrid Routing (Hardware-friendly substitute: Pruned Random Forest 5-subforest ensemble + ArcheType-Gemini LLM escalation)
  3. Pure RECA Ensemble (Local-only 5-member RECA + table-level soft penalty)
  4. Pure Random Forest Classifier (Local-only fast substitute for CPU/low-resource hardware)
"""

import os
import re
import json
import time
import random
import joblib
import requests
import numpy as np
import pandas as pd

from src.rf_features import (
    CharRatioExtractor,
    NumericalStatsExtractor,
    DateStatsExtractor,
    space_tokenizer,
)
from src.reca_model import RECAEnsemblePredictor, check_reca_available

# 23 Rare / Hard Epidemiological Domain Classes that trigger LLM escalation in Hybrid Mode
WEAK_CLASSES = {
    "Pathogen", "Pathogen_status", "Ethnicity", "Healthcare_worker", "Previous_infection",
    "Pregnancy_status", "Vaccine_name", "Date_onset", "Confirmation_method", "Hospitalised",
    "Intensive_care", "Date_isolation", "Date_death", "Contact_animal", "Travel_history",
    "Travel_history_entry", "Travel_history_start", "Travel_history_location",
    "Genomics_Metadata", "Date_entry", "Date_last_modified", "Recovered", "Name_of_family_head"
}

# Optimal Hyperparameter Presets discovered during thesis evaluation
HYPERPARAMETER_PRESETS = {
    "reca_pareto": {
        "name": "RECA Hybrid — Pareto Optimal (Thresh = 0.50, Tol = 2)",
        "backbone": "reca",
        "hybrid": True,
        "conf_threshold": 0.50,
        "ensemble_tolerance": 2,
        "use_weak_classes": True,
        "use_soft_penalty": True,
        "default_penalty": 0.05,
        "date_penalty": 0.10,
        "description": "Best efficiency knee-point for RECA: routes ~26.3% of columns to LLM while achieving 0.6869 Macro F1 / 0.7906 Micro F1.",
    },
    "reca_max_f1": {
        "name": "RECA Hybrid — Max Macro F1 (Thresh = 0.50, Tol = 1)",
        "backbone": "reca",
        "hybrid": True,
        "conf_threshold": 0.50,
        "ensemble_tolerance": 1,
        "use_weak_classes": True,
        "use_soft_penalty": True,
        "default_penalty": 0.05,
        "date_penalty": 0.10,
        "description": "Highest overall accuracy for RECA: requires unanimous 5-model consensus (Tol=1), routing ~49.6% to LLM for 0.7598 Macro F1 / 0.8471 Micro F1.",
    },
    "rf_pareto": {
        "name": "RF Hybrid (Hardware Substitute) — Pareto Optimal (Thresh = 0.30, Tol = 2)",
        "backbone": "rf",
        "hybrid": True,
        "conf_threshold": 0.30,
        "ensemble_tolerance": 2,
        "use_weak_classes": True,
        "use_soft_penalty": False,
        "default_penalty": 1.0,
        "date_penalty": 1.0,
        "description": "Best Pareto balance for lightweight hardware (CPU-only): 5 sub-forests with Thresh=0.30, Tol=2 (~44.6% routed to LLM, 0.6828 Macro F1 / 0.7820 Micro F1).",
    },
    "rf_max_f1": {
        "name": "RF Hybrid (Hardware Substitute) — Max Macro F1 (Thresh = 0.55, Tol = 1)",
        "backbone": "rf",
        "hybrid": True,
        "conf_threshold": 0.55,
        "ensemble_tolerance": 1,
        "use_weak_classes": True,
        "use_soft_penalty": False,
        "default_penalty": 1.0,
        "date_penalty": 1.0,
        "description": "Highest accuracy for RF Hybrid substitute: Thresh=0.55, Tol=1 (0.7636 Macro F1 / 0.8433 Micro F1).",
    },
    "pure_reca": {
        "name": "Pure RECA Ensemble (100% Local Deep Learning)",
        "backbone": "reca",
        "hybrid": False,
        "conf_threshold": 0.0,
        "ensemble_tolerance": 5,
        "use_weak_classes": False,
        "use_soft_penalty": True,
        "default_penalty": 0.05,
        "date_penalty": 0.10,
        "description": "Runs 100% locally using the 5-member fine-tuned multilingual BERT RECA ensemble + table-level soft-penalty heuristic.",
    },
    "pure_rf": {
        "name": "Pure Random Forest Substitute (100% Local, Fast CPU)",
        "backbone": "rf",
        "hybrid": False,
        "conf_threshold": 0.0,
        "ensemble_tolerance": 5,
        "use_weak_classes": False,
        "use_soft_penalty": False,
        "default_penalty": 1.0,
        "date_penalty": 1.0,
        "description": "Runs 100% locally in milliseconds on any CPU without GPU or PyTorch, using the pruned 10,004-dim Random Forest ensemble.",
    },
}


def load_vocab(base_dir=None):
    """Loads the 74 epidemiological semantic classes from type_vocab.txt."""
    candidates = []
    if base_dir:
        candidates.append(os.path.join(base_dir, "type_vocab.txt"))
    candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "type_vocab.txt")))
    candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RECA-paper", "finetune_headers", "fold_0", "type_vocab.txt")))

    vocab = {}
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        vocab[int(parts[0])] = " ".join(parts[1:])
            return vocab
    raise FileNotFoundError("Could not locate type_vocab.txt.")


def apply_soft_penalty(table_probs_matrix, vocab, default_penalty=0.05, date_penalty=0.10):
    """
    Applies the table-level greedy soft-penalty heuristic to discourage duplicate non-repeatable
    semantic classes within the same table:
      - Location_information, Cases, Deaths, Boolean, Symptom: penalty = 1.00 (no penalty)
      - Date_* classes: penalty = date_penalty (0.10)
      - All other classes: penalty = default_penalty (0.05)
    """
    probs = table_probs_matrix.copy()
    num_cols, num_classes = probs.shape
    allow_repeat_classes = {"Location_information", "Cases", "Deaths", "Boolean", "Symptom"}
    anchored_classes = set()

    all_probs = []
    for c in range(num_cols):
        for k in range(num_classes):
            all_probs.append((c, k, float(probs[c, k])))

    all_probs.sort(key=lambda x: x[2], reverse=True)

    for col_idx, class_idx, _ in all_probs:
        if class_idx not in anchored_classes:
            anchored_classes.add(class_idx)
            class_name = vocab.get(class_idx, "")
            if class_name.startswith("Date_"):
                current_penalty = date_penalty
            elif class_name in allow_repeat_classes:
                current_penalty = 1.00
            else:
                current_penalty = default_penalty

            if current_penalty < 1.0:
                for other_col in range(num_cols):
                    if other_col != col_idx:
                        probs[other_col, class_idx] *= current_penalty

    return probs


class RFEnsemblePredictor:
    """
    Runs the Pruned Random Forest Classifier (word_tf_idf + char_tf_idf + char_ratios)
    and partitions its decision trees into 5 disjoint sub-forests to compute:
      - Full probability distribution across the 74 semantic classes
      - Top-1 confidence score p_max
      - Number of distinct sub-forest answers N_diff in {1..5}
    """
    def __init__(self, base_dir=None, n_classes=74, fold_idx=0):
        self.base_dir = base_dir
        self.n_classes = n_classes
        self.fold_idx = fold_idx
        self.pipeline = None
        self._loaded = False

    def load(self):
        if self._loaded and self.pipeline is not None:
            return True

        import sys
        # Ensure custom feature extractors are resolvable under __main__ if unpickling legacy files
        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            for cls_name, cls_obj in [
                ("CharRatioExtractor", CharRatioExtractor),
                ("NumericalStatsExtractor", NumericalStatsExtractor),
                ("DateStatsExtractor", DateStatsExtractor),
                ("space_tokenizer", space_tokenizer),
            ]:
                if not hasattr(main_mod, cls_name):
                    setattr(main_mod, cls_name, cls_obj)

        candidates = []
        if self.base_dir:
            candidates.append(os.path.join(self.base_dir, "checkpoints_rf", f"rf_hybrid_fold_{self.fold_idx}.pkl"))
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints_rf", f"rf_hybrid_fold_{self.fold_idx}.pkl")))
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RECA-paper", "Semtab", "checkpoints_rf", f"rf_hybrid_fold_{self.fold_idx}.pkl")))

        for path in candidates:
            if os.path.isfile(path):
                self.pipeline = joblib.load(path)
                self._loaded = True
                return True

        raise FileNotFoundError("Random Forest checkpoint not found in 'checkpoints_rf/'.")

    def predict_columns(self, col_texts):
        self.load()
        features_step = self.pipeline.named_steps["features"]
        clf = self.pipeline.named_steps["clf"]
        classes = clf.classes_

        X_te = features_step.transform(col_texts)
        rf_proba_raw = clf.predict_proba(X_te)

        # Partition clf.estimators_ into 5 disjoint sub-forests to measure consensus N_diff (1..5)
        n_cols = len(col_texts)
        subforest_preds = np.zeros((n_cols, 5), dtype=int)
        for sf_idx in range(5):
            sub_trees = clf.estimators_[sf_idx::5]
            sub_prob = np.mean([t.predict_proba(X_te) for t in sub_trees], axis=0)
            sub_cls_idx = np.argmax(sub_prob, axis=1)
            subforest_preds[:, sf_idx] = classes[sub_cls_idx]

        results = []
        for idx in range(n_cols):
            full_probs = np.zeros(self.n_classes, dtype=np.float32)
            for c_pos, cls_id in enumerate(classes):
                if 0 <= cls_id < self.n_classes:
                    full_probs[cls_id] = rf_proba_raw[idx, c_pos]

            top1_idx = int(np.argmax(full_probs))
            top1_prob = float(np.max(full_probs))
            indiv_preds = [int(x) for x in subforest_preds[idx]]
            num_diff = len(set(indiv_preds))

            results.append({
                "probs": full_probs,
                "top1_idx": top1_idx,
                "top1_prob": top1_prob,
                "individual_preds": indiv_preds,
                "num_different_answers": num_diff,
            })
        return results


def query_archetype_llm_column(
    col_header,
    sample_values,
    other_col_samples,
    valid_labels,
    api_key=None,
    model_name="gemini-2.5-flash",
    ollama_url=None,
):
    """
    ArcheType-style LLM column annotation query for columns escalated by the Hybrid Router.
    Supports Google Gemini API (via GOOGLE_API_KEY or user-supplied key) or local Ollama fallback.
    """
    context_str = f"{col_header} : " + " ".join(sample_values[:50])
    labels_str = ", ".join(valid_labels)
    ocs_str = ""
    if other_col_samples:
        ocs_str = "For additional context, here are some entries from other columns in the table: " + ", ".join(other_col_samples[:10]) + "\n"

    prompt = (
        f"SYSTEM: Please select the class from {labels_str} which best describes the context. "
        f"Please respond only with the name of the class.\n"
        f"Context: {context_str}\n"
        f"{ocs_str}\n"
        f"Response:\n"
    )

    raw_answer = ""
    effective_key = api_key or os.environ.get("GOOGLE_API_KEY", "").strip()

    if effective_key:
        # Try standard Gemini generateContent endpoint first, then interactions endpoint
        try:
            gen_model = model_name if model_name else "gemini-2.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gen_model}:generateContent?key={effective_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0},
            }
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    raw_answer = "".join(p.get("text", "") for p in parts).strip()
        except Exception:
            pass

    if not raw_answer and ollama_url:
        try:
            url = ollama_url.rstrip("/") + "/api/generate"
            resp = requests.post(
                url,
                json={"model": model_name or "qwen2.5:7b", "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
                timeout=30,
            )
            if resp.status_code == 200:
                raw_answer = resp.json().get("response", "").strip()
        except Exception:
            pass

    if not raw_answer:
        return None

    # Match raw_answer to valid_labels
    clean_ans = raw_answer.strip("`'\"\n\r\t .")
    lower_map = {lbl.lower(): lbl for lbl in valid_labels}
    if clean_ans.lower() in lower_map:
        return lower_map[clean_ans.lower()]

    # Substring containment fallback (sorted by length descending)
    for lbl in sorted(valid_labels, key=len, reverse=True):
        if lbl.lower() in clean_ans.lower():
            return lbl

    return None


class HybridAnnotatorEngine:
    """
    Main annotation facade used by the Web GUI and CLI.
    Provides column-level predictions, confidence scores, consensus counts, routing triggers,
    and top-3 alternative candidate classes for easy one-click user overrides.
    """
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.vocab = load_vocab(self.base_dir)
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        self.valid_labels = [v for v in self.vocab.values() if v.lower() != "skip"]
        self.rf_predictor = RFEnsemblePredictor(base_dir=self.base_dir, n_classes=len(self.vocab), fold_idx=0)
        self.reca_predictor = RECAEnsemblePredictor(base_dir=self.base_dir, n_classes=len(self.vocab), data_fold=0)

    def get_hardware_status(self):
        reca_ok, reca_info, n_reca_ckpts = check_reca_available(self.base_dir)
        rf_ok = False
        try:
            self.rf_predictor.load()
            rf_ok = True
        except Exception:
            rf_ok = False

        return {
            "reca_available": reca_ok,
            "reca_device": reca_info,
            "reca_checkpoints": n_reca_ckpts,
            "rf_available": rf_ok,
            "llm_api_configured": bool(os.environ.get("GOOGLE_API_KEY", "").strip()),
            "presets": HYPERPARAMETER_PRESETS,
            "all_classes": sorted(self.valid_labels),
        }

    def annotate_dataframe(
        self,
        df,
        mode="reca_pareto",
        conf_threshold=None,
        ensemble_tolerance=None,
        use_weak_classes=None,
        use_soft_penalty=None,
        default_penalty=0.05,
        date_penalty=0.10,
        api_key=None,
        llm_model="gemini-2.5-flash",
    ):
        """
        Annotates all columns in `df`.
        If RECA is requested on hardware where RECA is unavailable, automatically falls back
        to the Random Forest substitute with its corresponding optimal hyperparameters and reports the fallback.
        """
        preset = HYPERPARAMETER_PRESETS.get(mode, HYPERPARAMETER_PRESETS["reca_pareto"]).copy()
        backbone = preset["backbone"]
        is_hybrid = preset["hybrid"]

        th = float(conf_threshold) if conf_threshold is not None else preset["conf_threshold"]
        tol = int(ensemble_tolerance) if ensemble_tolerance is not None else preset["ensemble_tolerance"]
        weak_active = bool(use_weak_classes) if use_weak_classes is not None else preset["use_weak_classes"]
        penalty_active = bool(use_soft_penalty) if use_soft_penalty is not None else preset["use_soft_penalty"]

        fallback_notice = None
        if backbone == "reca":
            reca_ok, reca_reason, _ = check_reca_available(self.base_dir)
            if not reca_ok:
                backbone = "rf"
                # Switch to RF optimal threshold if using default RECA threshold
                if conf_threshold is None:
                    th = 0.30 if tol >= 2 else 0.55
                penalty_active = False
                fallback_notice = (
                    f"RECA Ensemble unavailable on this hardware ({reca_reason}). "
                    f"Automatically switched to the Pruned Random Forest substitute (Thresh={th:.2f}, Tol={tol})."
                )

        # Build column text representations ("Header : val1 val2 ...")
        col_names = list(df.columns)
        col_texts = []
        col_sample_vals = []
        other_samples_all = []

        for col in col_names:
            vals = df[col].dropna().astype(str).tolist()
            sampled = vals[:500]
            col_sample_vals.append(sampled[:30])
            text_repr = f"{str(col)} : " + " ".join(sampled)
            col_texts.append(text_repr)
            if sampled:
                other_samples_all.append(f"{str(col)} : " + " ".join(sampled[:5]))

        # 1. Run local ensemble (RECA or RF)
        if backbone == "reca":
            base_outputs = self.reca_predictor.predict_columns(col_texts)
        else:
            base_outputs = self.rf_predictor.predict_columns(col_texts)

        n_cols = len(col_names)
        n_classes = len(self.vocab)
        hybrid_probs_matrix = np.zeros((n_cols, n_classes), dtype=np.float32)

        routing_details = []
        routed_llm_count = 0
        routed_local_count = 0

        for idx, res in enumerate(base_outputs):
            probs = res["probs"].copy()
            # Mask out "Skip" class (index 7) so it is never predicted
            if 7 in self.vocab and self.vocab[7].lower() == "skip":
                probs[7] = 0.0

            top1_idx = int(np.argmax(probs))
            top1_prob = float(np.max(probs))
            num_diff = int(res["num_different_answers"])
            candidate_class = self.vocab.get(top1_idx, "Unknown")

            triggers = []
            if num_diff > tol:
                triggers.append(f"Disagreement (N_diff={num_diff} > {tol})")
            if top1_prob < th:
                triggers.append(f"Low Confidence ({top1_prob:.2f} < {th:.2f})")
            if weak_active and candidate_class in WEAK_CLASSES:
                triggers.append(f"Weak Domain Class ({candidate_class})")

            needs_escalation = is_hybrid and (len(triggers) > 0)
            routed_to = "RECA Ensemble" if backbone == "reca" else "Random Forest"
            llm_overrode = False

            if needs_escalation:
                other_ctx = [s for j, s in enumerate(other_samples_all) if j != idx]
                llm_label = query_archetype_llm_column(
                    col_header=str(col_names[idx]),
                    sample_values=col_sample_vals[idx],
                    other_col_samples=other_ctx,
                    valid_labels=self.valid_labels,
                    api_key=api_key,
                    model_name=llm_model,
                )
                if llm_label and llm_label in self.rev_vocab:
                    llm_idx = self.rev_vocab[llm_label]
                    h_probs = np.zeros(n_classes, dtype=np.float32)
                    h_probs[llm_idx] = 1.0
                    hybrid_probs_matrix[idx] = h_probs
                    routed_to = "ArcheType-LLM (Hybrid)"
                    routed_llm_count += 1
                    llm_overrode = True
                else:
                    # If no API key is configured and local model is RECA, fall back to RF ensemble if RF is more confident, or keep local probs
                    if backbone == "reca":
                        try:
                            rf_res = self.rf_predictor.predict_columns([col_texts[idx]])[0]
                            rf_probs = rf_res["probs"]
                            if 7 in self.vocab:
                                rf_probs[7] = 0.0
                            # Blend RECA and RF probabilities when offline
                            blended = 0.65 * probs + 0.35 * rf_probs
                            hybrid_probs_matrix[idx] = blended
                            routed_to = "Hybrid Triggered (Local Ensemble Fallback)"
                        except Exception:
                            hybrid_probs_matrix[idx] = probs
                            routed_to = "Hybrid Triggered (Retained Local)"
                    else:
                        hybrid_probs_matrix[idx] = probs
                        routed_to = "Hybrid Triggered (Retained Local RF)"
                    routed_local_count += 1
            else:
                hybrid_probs_matrix[idx] = probs
                routed_local_count += 1

            routing_details.append({
                "col_index": idx,
                "original_header": str(col_names[idx]),
                "local_candidate": candidate_class,
                "confidence": round(top1_prob, 4),
                "num_different_answers": num_diff,
                "ensemble_agreement": f"{6 - num_diff}/5" if num_diff <= 5 else "1/5",
                "triggers": triggers,
                "needs_escalation": needs_escalation,
                "routed_to": routed_to,
                "llm_overrode": llm_overrode,
                "raw_probs": probs,
            })

        # 2. Apply Table-Level Soft Penalty (active for RECA modes, off for RF modes per thesis findings)
        if penalty_active and n_cols > 1:
            final_probs_matrix = apply_soft_penalty(
                hybrid_probs_matrix,
                self.vocab,
                default_penalty=default_penalty,
                date_penalty=date_penalty,
            )
        else:
            final_probs_matrix = hybrid_probs_matrix

        final_idxs = np.argmax(final_probs_matrix, axis=1)

        columns_report = []
        for idx in range(n_cols):
            final_idx = int(final_idxs[idx])
            final_label = self.vocab.get(final_idx, "Unknown")
            raw_p = routing_details[idx]["raw_probs"]
            top3_indices = np.argsort(raw_p)[::-1][:5]
            alternatives = []
            for alt_i in top3_indices:
                alt_name = self.vocab.get(int(alt_i), "Unknown")
                if alt_name.lower() == "skip":
                    continue
                alternatives.append({
                    "label": alt_name,
                    "prob": round(float(raw_p[alt_i]), 4),
                })
                if len(alternatives) >= 4:
                    break

            detail = routing_details[idx]
            columns_report.append({
                "col_index": idx,
                "original_header": detail["original_header"],
                "predicted_label": final_label,
                "local_candidate": detail["local_candidate"],
                "confidence": detail["confidence"],
                "num_different_answers": detail["num_different_answers"],
                " triggers": detail["triggers"],
                "triggers": detail["triggers"],
                "needs_escalation": detail["needs_escalation"],
                "routed_to": detail["routed_to"],
                "alternatives": alternatives,
            })

        symptoms_cols = [
            c["original_header"]
            for c in columns_report
            if c["predicted_label"] == "Symptoms" or c["original_header"].strip().lower() == "symptoms"
        ]

        return {
            "backbone_used": backbone,
            "hybrid_enabled": is_hybrid,
            "conf_threshold": th,
            "ensemble_tolerance": tol,
            "use_weak_classes": weak_active,
            "use_soft_penalty": penalty_active,
            "fallback_notice": fallback_notice,
            "routed_llm_count": routed_llm_count,
            "routed_local_count": routed_local_count,
            "columns": columns_report,
            "detected_symptoms_columns": symptoms_cols,
        }
