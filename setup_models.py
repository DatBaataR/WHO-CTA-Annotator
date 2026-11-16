"""
setup_models.py — Model & Checkpoint Setup Utility for WHO-CTA-Annotator.

Because GitHub repositories have file size limits, the repository comes bundled with
the lightweight 5-fold Pruned Random Forest checkpoints (~14 MB total) so the project
runs out-of-the-box on any CPU hardware.

Run this script to download / install the deep learning models for full RECA Ensemble
and Multilingual Medical NER Symptom Extraction:
  1. SpaCy English lemmatization dictionary (`en_core_web_sm`)
  2. HUMADEX/english_medical_ner -> `./models/english_medical_ner`
  3. facebook/m2m100_418M -> `./models/m2m100_418M`
  4. bert-base-multilingual-uncased (cached for RECA KREL backbone)
  5. RECA 5-fold fine-tuned checkpoints -> `./checkpoints_reca/`
"""

import os
import sys
import shutil
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_spacy_model():
    print("\n[1/4] Checking SpaCy 'en_core_web_sm' dictionary...")
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("  -> 'en_core_web_sm' is already installed.")
    except Exception:
        print("  -> Downloading 'en_core_web_sm'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


def download_hf_models(skip_translator=False):
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        AutoModelForSeq2SeqLM,
        BertTokenizer,
        BertModel,
    )

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Medical NER
    ner_target = os.path.join(models_dir, "english_medical_ner")
    print(f"\n[2/4] Setting up Medical NER model (HUMADEX/english_medical_ner) -> {ner_target}")
    if os.path.isdir(ner_target) and os.path.isfile(os.path.join(ner_target, "config.json")):
        print("  -> Already present locally.")
    else:
        tok = AutoTokenizer.from_pretrained("HUMADEX/english_medical_ner")
        mod = AutoModelForTokenClassification.from_pretrained("HUMADEX/english_medical_ner")
        tok.save_pretrained(ner_target)
        mod.save_pretrained(ner_target)
        print("  -> Saved HUMADEX/english_medical_ner.")

    # 2. M2M100 Translator
    if not skip_translator:
        trans_target = os.path.join(models_dir, "m2m100_418M")
        print(f"\n[3/4] Setting up Multilingual Translator (facebook/m2m100_418M) -> {trans_target}")
        if os.path.isdir(trans_target) and os.path.isfile(os.path.join(trans_target, "config.json")):
            print("  -> Already present locally.")
        else:
            tok = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
            mod = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")
            tok.save_pretrained(trans_target)
            mod.save_pretrained(trans_target)
            print("  -> Saved facebook/m2m100_418M.")
    else:
        print("\n[3/4] Skipping M2M100 translator download (--skip-translator specified).")

    # 3. Cache bert-base-multilingual-uncased for RECA
    print("\n[4/4] Caching 'bert-base-multilingual-uncased' for RECA Ensemble...")
    BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    BertModel.from_pretrained("bert-base-multilingual-uncased")
    print("  -> Cached bert-base-multilingual-uncased.")


def setup_reca_checkpoints(source_dir=None):
    reca_dir = os.path.join(BASE_DIR, "checkpoints_reca")
    os.makedirs(reca_dir, exist_ok=True)

    # Check if local sibling repo has the fine-tuned checkpoints
    default_sibling = os.path.abspath(os.path.join(BASE_DIR, "..", "RECA-paper", "Semtab", "checkpoints_new_labels"))
    src = source_dir or (default_sibling if os.path.isdir(default_sibling) else None)

    if src and os.path.isdir(src):
        print(f"\nCopying 5-fold RECA Ensemble checkpoints from {src} to {reca_dir}...")
        copied = 0
        for orig_fold in range(1, 6):
            fname = f"semtab-RECA-finetune_dataFold_0_origFold_{orig_fold}.pkl"
            s_path = os.path.join(src, fname)
            d_path = os.path.join(reca_dir, fname)
            if os.path.isfile(s_path) and not os.path.isfile(d_path):
                shutil.copy2(s_path, d_path)
                copied += 1
        print(f"  -> {copied} RECA checkpoint(s) copied into checkpoints_reca/.")
    else:
        existing = [f for f in os.listdir(reca_dir) if f.endswith(".pkl")]
        if existing:
            print(f"\nFound {len(existing)} RECA checkpoint(s) in {reca_dir}.")
        else:
            print(
                f"\nNote: Place your fine-tuned RECA checkpoint files "
                f"(semtab-RECA-finetune_dataFold_0_origFold_1..5.pkl) into:\n  {reca_dir}\n"
                f"Until then, the application will automatically use the bundled Random Forest substitute."
            )


def main():
    parser = argparse.ArgumentParser(description="Download and configure NLP & RECA models for WHO-CTA-Annotator")
    parser.add_argument("--skip-translator", action="store_true", help="Skip downloading the 1.6GB M2M100 translation model")
    parser.add_argument("--copy-reca-from", type=str, default=None, help="Optional path to copy RECA .pkl checkpoints from")
    args = parser.parse_args()

    download_spacy_model()
    download_hf_models(skip_translator=args.skip_translator)
    if args.copy_reca_from:
        setup_reca_checkpoints(source_dir=args.copy_reca_from)
    else:
        print(
            "\n[RECA Checkpoints] To keep the repository lightweight for GitHub, RECA .pkl files are not copied by default.\n"
            "If you want to copy the 5 fold-0 RECA checkpoints locally, run:\n"
            "  python setup_models.py --copy-reca-from ../RECA-paper/Semtab/checkpoints_new_labels"
        )
    print("\nSetup complete! Start the Web GUI with:\n  python app.py")


if __name__ == "__main__":
    main()
