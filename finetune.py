"""
finetune.py — Continual Fine-Tuning & Retraining Script for WHO-CTA-Annotator.

Allows WHO employees to continually fine-tune the existing 5-member RECA Ensemble
(and/or retrain the Random Forest Substitute) on newly annotated epidemiological data
without needing the original 275-class SemTab pre-trained checkpoints.

Why continual fine-tuning on existing WHO RECA checkpoints works (and is preferred):
  - The original SemTab checkpoints had 275 Wikipedia/DBpedia classes, requiring the
    classification heads (`fcc_tar`, `fcc_rel`, `fcc_sub`) to be discarded and randomly
    re-initialized.
  - The existing WHO-finetuned RECA checkpoints (`checkpoints_reca/semtab-RECA-finetune_dataFold_0_origFold_1..5.pkl`)
    already have both a domain-adapted BERT encoder AND a warm-started 74-class
    epidemiological classification head.
  - Continually fine-tuning from these 5 checkpoints with a gentle learning rate (e.g. 5e-6)
    preserves prior knowledge over all 74 WHO classes while adapting to new datasets.

Supported Input Training Data Formats (`--train-data`):
  Format A — Folder of Annotated CSV Tables (Easiest for WHO users!):
    Point `--train-data` to a directory containing one or more CSV tables whose column
    headers have already been annotated with valid WHO class names from `type_vocab.txt`
    (for example, CSVs exported directly from the WHO-CTA-Annotator Web GUI!).
    - Any column whose header matches a class in `type_vocab.txt` (case-insensitive)
      or has format `OriginalName (ClassName)` is automatically converted into a training sample.
    - Columns with `symptom: <name>` headers or unknown headers are safely ignored unless mapped.

  Format B — Pre-formatted Column-Level CSV File:
    Point `--train-data` to a single `.csv` file with columns:
      - `label`: either the integer class ID (0..73) or the exact class name (e.g. `Date_onset`, `Age`)
      - `data` : serialized column string `<OriginalHeader> : <val_1> <val_2> ...`
      - (Optional) `table_id`, `column_index`

Usage Examples:
  # 1. Continually fine-tune the 5 RECA checkpoints on a folder of annotated WHO CSV tables:
  python finetune.py --train-data ./new_training_data --model reca --epochs 10 --lr 5e-6

  # 2. Fine-tune both RECA and the Random Forest substitute on a column-level CSV:
  python finetune.py --train-data ./new_training_data/columns_train.csv --model both

  # 3. Overwrite checkpoints in-place so the Web GUI immediately uses the updated models:
  python finetune.py --train-data ./new_training_data --model both --in-place
"""

import os
import re
import sys
import time
import glob
import random
import shutil
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.hybrid_router import load_vocab
from src.reca_model import KREL, find_reca_checkpoint_files, HAS_TORCH
from src.rf_features import build_pruned_rf_pipeline

MAX_LEN = 512


def setup_seed(seed=20):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True


def resolve_label_to_id(raw_label, vocab, lower_to_id):
    """Maps either an integer ID ('4') or class name ('Adm1' or 'District (Adm1)') to 0..73."""
    s = str(raw_label).strip()
    if not s:
        return None
    # Check integer ID
    if s.isdigit():
        idx = int(s)
        if idx in vocab:
            return idx
    # Direct case-insensitive class name match
    if s.lower() in lower_to_id:
        return lower_to_id[s.lower()]
    # Check 'OriginalHeader (ClassName)' export format from Web GUI
    m = re.search(r"\(([^()]+)\)\s*$", s)
    if m:
        inner = m.group(1).strip().lower()
        if inner in lower_to_id:
            return lower_to_id[inner]
    # Check 'symptom: <name>' -> maps to 'Symptom' (ID 57) if present
    if s.lower().startswith("symptom:") and "symptom" in lower_to_id:
        return lower_to_id["symptom"]
    return None


def load_training_records(data_path, vocab, max_cells_per_col=500):
    """
    Loads training samples from either:
      - A column-level CSV file containing ('label', 'data') columns
      - A single annotated WHO table CSV
      - A directory of annotated WHO table CSVs (or column-level CSVs)
    Returns a DataFrame with columns: ['table_id', 'column_index', 'label', 'label_name', 'data']
    """
    lower_to_id = {name.lower(): idx for idx, name in vocab.items()}
    csv_files = []

    if os.path.isfile(data_path):
        csv_files = [data_path]
    elif os.path.isdir(data_path):
        csv_files = sorted(glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True))
    else:
        raise FileNotFoundError(f"Training data path not found: {data_path}")

    if not csv_files:
        raise ValueError(f"No .csv files found in {data_path}")

    records = []
    skipped_cols = []

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath, dtype=str)
        except Exception as e:
            print(f"  [WARN] Could not read {fname}: {e}")
            continue

        # Check if this CSV is already in pre-formatted column-level schema ('label' and 'data')
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        if "label" in cols_lower and "data" in cols_lower and len(df.columns) <= 6:
            lbl_col = cols_lower["label"]
            dat_col = cols_lower["data"]
            tid_col = cols_lower.get("table_id", None)
            cidx_col = cols_lower.get("column_index", None)

            for row_i, row in df.iterrows():
                lid = resolve_label_to_id(row[lbl_col], vocab, lower_to_id)
                if lid is None:
                    skipped_cols.append(f"{fname}#row{row_i} ('{row[lbl_col]}')")
                    continue
                text_val = str(row[dat_col]) if pd.notna(row[dat_col]) else "[PAD]"
                records.append({
                    "table_id": str(row[tid_col]) if tid_col else fname,
                    "column_index": int(row[cidx_col]) if (cidx_col and str(row[cidx_col]).isdigit()) else row_i,
                    "label": int(lid),
                    "label_name": vocab[int(lid)],
                    "data": text_val,
                })
        else:
            # Treat as an annotated WHO outbreak table CSV where column headers are WHO semantic types
            for col_idx, col_header in enumerate(df.columns):
                lid = resolve_label_to_id(col_header, vocab, lower_to_id)
                if lid is None:
                    skipped_cols.append(f"{fname}:'{col_header}'")
                    continue
                vals = df.iloc[:, col_idx].dropna().astype(str).tolist()[:max_cells_per_col]
                val_str = " ".join(vals) if vals else "[PAD]"
                # If header was 'OrigName (PredClass)', keep OrigName as prefix in data string
                m = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", str(col_header).strip())
                orig_h = m.group(1).strip() if m else str(col_header).strip()
                col_text = f"{orig_h} : {val_str}"

                records.append({
                    "table_id": os.path.splitext(fname)[0],
                    "column_index": col_idx,
                    "label": int(lid),
                    "label_name": vocab[int(lid)],
                    "data": col_text,
                })

    if not records:
        raise ValueError(
            "No valid labeled columns were found! Ensure your CSV headers match class names in "
            "'type_vocab.txt' (or provide a CSV with 'label' and 'data' columns)."
        )

    out_df = pd.DataFrame(records)
    print(f"\nLoaded {len(out_df)} labeled training column(s) across {out_df['table_id'].nunique()} table(s) ({out_df['label'].nunique()} distinct WHO classes).")
    if skipped_cols:
        preview_skips = ", ".join(skipped_cols[:8])
        more = f" ... (+{len(skipped_cols) - 8} more)" if len(skipped_cols) > 8 else ""
        print(f"  [Note] Skipped {len(skipped_cols)} unrecognized column(s): {preview_skips}{more}")
    return out_df


def finetune_reca_ensemble(
    train_df,
    val_df,
    vocab,
    output_dir,
    epochs=10,
    lr=5e-6,
    batch_size=8,
    early_stop_patience=4,
    freeze_bert_layers=0,
):
    """
    Continually fine-tunes the 5 RECA ensemble members in `checkpoints_reca/`
    (or initializes from base `bert-base-multilingual-uncased` if no checkpoints exist yet).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch and Transformers are required to fine-tune the RECA Ensemble.")

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer
    from tqdm import tqdm

    class ColumnDataset(Dataset):
        def __init__(self, df_sub, tokenizer, max_len=512):
            self.df = df_sub.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            text = str(row["data"]) if pd.notna(row["data"]) else "[PAD]"
            label = int(row["label"])
            tok = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
            )
            ids = torch.tensor(tok["input_ids"], dtype=torch.long)
            return ids, ids, ids, torch.tensor(label, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[RECA Fine-Tuning] Device: {device} | Epochs: {epochs} | LR: {lr} | Batch Size: {batch_size}")

    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased", local_files_only=True)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

    existing_ckpts = find_reca_checkpoint_files(BASE_DIR, data_fold=0)
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(vocab)

    for member_idx in range(1, 6):
        print("-" * 70)
        print(f"Fine-Tuning RECA Ensemble Member {member_idx}/5")
        print("-" * 70)

        # Bootstrap/shuffle slightly per member if no validation split was provided
        m_train_df = train_df.sample(frac=1.0, random_state=20 + member_idx).reset_index(drop=True)
        train_loader = DataLoader(ColumnDataset(m_train_df, tokenizer, MAX_LEN), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ColumnDataset(val_df, tokenizer, MAX_LEN), batch_size=batch_size, shuffle=False) if (val_df is not None and len(val_df) > 0) else None

        model = KREL(n_classes=n_classes)

        # Load existing fine-tuned WHO RECA checkpoint (or compatible pre-trained checkpoint)
        src_ckpt = existing_ckpts[member_idx - 1] if (member_idx - 1) < len(existing_ckpts) else None
        if src_ckpt and os.path.isfile(src_ckpt):
            print(f"  Loading warm-start checkpoint: {src_ckpt}")
            sd = torch.load(src_ckpt, map_location="cpu")
            model_dict = model.state_dict()
            # Keep all matching tensors (preserves 74-class head if shapes match, or drops head if 275-class SemTab checkpoint)
            compatible_sd = {k: v for k, v in sd.items() if k in model_dict and v.shape == model_dict[k].shape}
            missing_heads = set(model_dict.keys()) - set(compatible_sd.keys())
            model.load_state_dict(compatible_sd, strict=False)
            if missing_heads:
                print(f"  [Info] Re-initialized classification head layers due to class count change: {sorted(missing_heads)}")
        else:
            print("  [Warning] No existing RECA checkpoint found for this member; fine-tuning from base mBERT.")

        # Optional layer freezing for very small datasets to prevent overfitting
        if freeze_bert_layers > 0:
            for layer_i in range(min(freeze_bert_layers, 12)):
                for param in model.bert_model.encoder.layer[layer_i].parameters():
                    param.requires_grad = False
            print(f"  Froze bottom {freeze_bert_layers} BERT encoder layer(s).")

        model = model.to(device)

        weight_decay = 1e-2
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

        best_val_loss = float("inf")
        no_improve = 0
        save_path = os.path.join(output_dir, f"semtab-RECA-finetune_dataFold_0_origFold_{member_idx}.pkl")

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            steps = 0

            for ids, rels, subs, labels in tqdm(train_loader, desc=f"Member {member_idx} | Epoch {epoch}/{epochs}", leave=False):
                ids = ids.to(device)
                rels = rels.to(device)
                subs = subs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(ids, rels, subs)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                steps += 1

            avg_train_loss = total_loss / max(1, steps)

            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                preds_all, trues_all = [], []
                with torch.no_grad():
                    for ids, rels, subs, labels in val_loader:
                        ids = ids.to(device)
                        rels = rels.to(device)
                        subs = subs.to(device)
                        labels = labels.to(device)
                        logits = model(ids, rels, subs)
                        vloss = loss_fn(logits, labels)
                        val_loss += vloss.item()
                        val_steps += 1
                        preds_all.append(logits.argmax(dim=1).cpu().numpy())
                        trues_all.append(labels.cpu().numpy())

                avg_val_loss = val_loss / max(1, val_steps)
                preds_cat = np.concatenate(preds_all)
                trues_cat = np.concatenate(trues_all)
                w_f1 = f1_score(trues_cat, preds_cat, average="weighted", zero_division=0)
                m_f1 = f1_score(trues_cat, preds_cat, average="macro", zero_division=0)
                print(f"  Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Weighted F1: {w_f1:.4f} | Val Macro F1: {m_f1:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), save_path)
                    no_improve = 0
                    print(f"    -> Saved improved checkpoint to {save_path}")
                else:
                    no_improve += 1
                    if no_improve >= early_stop_patience:
                        print(f"    -> Early stopping triggered after {epoch} epochs.")
                        break
            else:
                print(f"  Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f}")
                torch.save(model.state_dict(), save_path)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[RECA Fine-Tuning Complete] Saved 5 ensemble checkpoints to: {output_dir}")


def retrain_rf_substitute(
    new_df,
    output_dir,
    combine_with_existing_Csv=None,
):
    """
    Trains and compresses the 5-fold Pruned Random Forest substitute models (`rf_hybrid_fold_0..4.pkl`).
    """
    os.makedirs(output_dir, exist_ok=True)
    combined_df = new_df.copy()

    if combine_with_existing_Csv and os.path.isfile(combine_with_existing_Csv):
        print(f"\n[RF Training] Merging new data with base training CSV: {combine_with_existing_Csv}")
        base_df = pd.read_csv(combine_with_existing_Csv, dtype=str)
        base_df["label"] = base_df["label"].astype(int)
        base_df["data"] = base_df["data"].fillna("")
        combined_df = pd.concat([base_df[["label", "data"]], combined_df[["label", "data"]]], ignore_index=True)

    X_all = combined_df["data"].fillna("").tolist()
    y_all = combined_df["label"].astype(int).values

    print(f"\n[RF Substitute Training] Fitting 5 Pruned Random Forest models on {len(X_all)} columns...")
    for fold_idx in range(5):
        pipeline = build_pruned_rf_pipeline(fold_idx=fold_idx, random_state=42 + fold_idx)
        pipeline.fit(X_all, y_all)
        out_path = os.path.join(output_dir, f"rf_hybrid_fold_{fold_idx}.pkl")
        joblib.dump(pipeline, out_path, compress=3)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  -> Saved fold {fold_idx} RF checkpoint: {out_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Continually fine-tune RECA Ensemble checkpoints and/or retrain the Random Forest substitute on new WHO data."
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to a folder of annotated WHO CSV tables, a single annotated table CSV, or a column-level CSV with 'label' and 'data' columns.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Optional path to validation data (same format as --train-data). If omitted and dataset has >= 20 columns, 15%% is used for validation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["reca", "rf", "both"],
        default="reca",
        help="Which model family to fine-tune/train: 'reca' (default), 'rf', or 'both'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of fine-tuning epochs for RECA (default: 8).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate for continual RECA fine-tuning (default: 5e-6; use 1e-5 if training from SemTab/scratch).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for RECA fine-tuning (default: 8).",
    )
    parser.add_argument(
        "--freeze-bert-layers",
        type=int,
        default=0,
        help="Optional number of bottom BERT encoder layers (0..12) to freeze when fine-tuning on very small datasets (default: 0).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite checkpoints directly inside checkpoints_reca/ and checkpoints_rf/ (backs up previous checkpoints to *.bak).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for fine-tuned checkpoints (defaults to checkpoints_reca_finetuned/ unless --in-place is set).",
    )
    args = parser.parse_args()

    setup_seed(20)
    vocab = load_vocab(BASE_DIR)

    full_df = load_training_records(args.train_data, vocab)

    if args.val_data:
        val_df = load_training_records(args.val_data, vocab)
        train_df = full_df
    elif len(full_df) >= 20:
        train_df, val_df = train_test_split(full_df, test_size=0.15, random_state=42)
        print(f"Split into {len(train_df)} train columns and {len(val_df)} validation columns.")
    else:
        train_df = full_df
        val_df = None
        print(f"Using all {len(train_df)} columns for training (dataset < 20 columns; validation split skipped).")

    if args.model in ("reca", "both"):
        if args.in_place:
            reca_out = os.path.join(BASE_DIR, "checkpoints_reca")
        else:
            reca_out = args.output_dir or os.path.join(BASE_DIR, "checkpoints_reca_finetuned")

        finetune_reca_ensemble(
            train_df=train_df,
            val_df=val_df,
            vocab=vocab,
            output_dir=reca_out,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            freeze_bert_layers=args.freeze_bert_layers,
        )

    if args.model in ("rf", "both"):
        if args.in_place:
            rf_out = os.path.join(BASE_DIR, "checkpoints_rf")
        else:
            rf_out = args.output_dir or os.path.join(BASE_DIR, "checkpoints_rf_finetuned")

        retrain_rf_substitute(
            new_df=full_df,
            output_dir=rf_out,
        )


if __name__ == "__main__":
    main()

