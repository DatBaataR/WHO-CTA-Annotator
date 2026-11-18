"""
setup_models.py — Model & Checkpoint Setup Utility for WHO-CTA-Annotator.

Because GitHub repositories have file size limits, the repository comes bundled with
the lightweight 5-fold Pruned Random Forest checkpoints (~14.3 MB total) so the project
runs out-of-the-box on any CPU hardware.

Run this script to download / install the deep learning models for full RECA Ensemble
and Multilingual Medical NER Symptom Extraction:
  1. SpaCy English lemmatization dictionary (`en_core_web_sm`)
  2. HUMADEX/english_medical_ner -> `./models/english_medical_ner`
  3. facebook/m2m100_418M -> `./models/m2m100_418M`
  4. bert-base-multilingual-uncased (cached for RECA KREL backbone)
  5. RECA 5-member fine-tuned checkpoints -> `./checkpoints_reca/` (from Google Drive or local folder)
"""

import os
import sys
import zipfile
import shutil
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paste your shared Google Drive folder or zip link here before pushing to GitHub if desired:
GOOGLE_DRIVE_RECA_URL = ""


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


def extract_any_zips_in_dir(target_dir):
    """Automatically extracts any .zip files placed inside checkpoints_reca/."""
    for fname in os.listdir(target_dir):
        if fname.lower().endswith(".zip"):
            zpath = os.path.join(target_dir, fname)
            print(f"  -> Extracting archive {fname} in {target_dir}...")
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(target_dir)


def download_reca_from_gdrive(gdrive_url, reca_dir):
    """Downloads RECA checkpoints from a Google Drive folder or file link using gdown."""
    try:
        import gdown
    except ImportError:
        print("  -> Installing 'gdown' to download from Google Drive...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    print(f"\nDownloading RECA checkpoints from Google Drive into {reca_dir}...")
    if "/folders/" in gdrive_url or "usp=drive_link" in gdrive_url and "/file/" not in gdrive_url:
        gdown.download_folder(url=gdrive_url, output=reca_dir, quiet=False, use_cookies=False)
    else:
        gdown.download(url=gdrive_url, output=reca_dir + os.sep, quiet=False, fuzzy=True)

    extract_any_zips_in_dir(reca_dir)


def setup_reca_checkpoints(source_dir=None, gdrive_url=None):
    reca_dir = os.path.join(BASE_DIR, "checkpoints_reca")
    os.makedirs(reca_dir, exist_ok=True)

    extract_any_zips_in_dir(reca_dir)

    effective_url = gdrive_url or GOOGLE_DRIVE_RECA_URL
    if effective_url:
        download_reca_from_gdrive(effective_url, reca_dir)

    if source_dir and os.path.isdir(source_dir):
        print(f"\nCopying 5-member RECA Ensemble checkpoints from {source_dir} to {reca_dir}...")
        copied = 0
        for orig_fold in range(1, 6):
            fname = f"semtab-RECA-finetune_dataFold_0_origFold_{orig_fold}.pkl"
            s_path = os.path.join(source_dir, fname)
            d_path = os.path.join(reca_dir, fname)
            if os.path.isfile(s_path) and not os.path.isfile(d_path):
                shutil.copy2(s_path, d_path)
                copied += 1
        print(f"  -> {copied} RECA checkpoint(s) copied into checkpoints_reca/.")

    from src.reca_model import find_reca_checkpoint_files
    found = find_reca_checkpoint_files(BASE_DIR)
    if found:
        print(f"\n[RECA Ready] Found {len(found)} RECA checkpoint(s) in checkpoints_reca/:")
        for p in found:
            print(f"  - {os.path.basename(p)}")
    else:
        print(
            "\n" + "=" * 75 + "\n"
            "  RECA Checkpoints (Google Drive):\n"
            "  Download the 5 fine-tuned RECA checkpoint files from Google Drive:\n"
            "    - semtab-RECA-finetune_dataFold_0_origFold_1.pkl\n"
            "    - semtab-RECA-finetune_dataFold_0_origFold_2.pkl\n"
            "    - semtab-RECA-finetune_dataFold_0_origFold_3.pkl\n"
            "    - semtab-RECA-finetune_dataFold_0_origFold_4.pkl\n"
            "    - semtab-RECA-finetune_dataFold_0_origFold_5.pkl\n"
            f"  and place them inside:\n    {reca_dir}\n\n"
            "  (Until they are placed there, the app automatically uses the bundled\n"
            "   5-Fold Pruned Random Forest substitute in checkpoints_rf/.)\n"
            + "=" * 75
        )


def main():
    parser = argparse.ArgumentParser(description="Download and configure NLP & RECA models for WHO-CTA-Annotator")
    parser.add_argument("--skip-translator", action="store_true", help="Skip downloading the 1.6GB M2M100 translation model")
    parser.add_argument("--gdrive-url", type=str, default=None, help="Google Drive link (folder or zip) to download RECA checkpoints into checkpoints_reca/")
    parser.add_argument("--copy-reca-from", type=str, default=None, help="Optional local path to copy RECA .pkl checkpoints from")
    args = parser.parse_args()

    download_spacy_model()
    download_hf_models(skip_translator=args.skip_translator)
    setup_reca_checkpoints(source_dir=args.copy_reca_from, gdrive_url=args.gdrive_url)
    print("\nSetup complete! Start the Web GUI by double-clicking WHO-CTA-Annotator.exe or running:\n  python app.py")


if __name__ == "__main__":
    main()
