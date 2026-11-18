"""
RECA (Relation-aware Entity and Column Annotation) KREL Architecture & Ensemble Inference.
Supports:
  - 5-fold or 25-checkpoint ensemble logit aggregation + softmax probability estimation
  - Extraction of per-member votes (N_diff) for hybrid disagreement routing
  - Automatic fallback to CPU if CUDA is unavailable
"""

import os
import ssl
import urllib3
import requests
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import torch
    from transformers import BertTokenizer, BertModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class KREL(torch.nn.Module):
        def __init__(self, n_classes=74, dim_k=768, dim_v=768, bert_name="bert-base-multilingual-uncased"):
            super(KREL, self).__init__()
            self.model_name = "KREL"
            try:
                self.bert_model = BertModel.from_pretrained(bert_name, local_files_only=True)
            except Exception:
                self.bert_model = BertModel.from_pretrained(bert_name)
            self.dropout = torch.nn.Dropout(0.3)
            self.fcc_tar = torch.nn.Linear(768, n_classes)
            self.fcc_rel = torch.nn.Linear(768, n_classes)
            self.fcc_sub = torch.nn.Linear(768, n_classes)
            self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(1)) for _ in range(3)])

        def encode(self, target_ids, rel_ids, sub_ids):
            att_tar = (target_ids > 0)
            _, tar = self.bert_model(input_ids=target_ids, attention_mask=att_tar, return_dict=False)
            att_rel = (rel_ids > 0)
            _, rel = self.bert_model(input_ids=rel_ids, attention_mask=att_rel, return_dict=False)
            att_sub = (sub_ids > 0)
            _, sub = self.bert_model(input_ids=sub_ids, attention_mask=att_sub, return_dict=False)
            return tar, rel, sub

        def forward(self, tar_ids, rel_ids, sub_ids):
            tar, rel, sub = self.encode(tar_ids, rel_ids, sub_ids)
            tar_out = self.dropout(tar)
            rel_out = self.dropout(rel)
            sub_out = self.dropout(sub)
            out_tar = self.fcc_tar(tar_out)
            out_rel = self.fcc_rel(rel_out)
            out_sub = self.fcc_sub(sub_out)
            res = self.weights[0] * out_tar + self.weights[1] * out_rel + self.weights[2] * out_sub
            return res
else:
    KREL = None


def find_reca_checkpoint_files(base_dir=None, data_fold=0):
    """
    Searches for RECA checkpoint files (.pkl) in:
      1. <base_dir>/checkpoints_reca/
      2. Sibling repo G:\CTA\RECA-paper\Semtab\checkpoints_new_labels/ (if present)
    For inference on new unseen tables, defaults to the 5 ensemble members of fold 0
    (or all available checkpoints if specified).
    """
    candidate_dirs = []
    if base_dir:
        candidate_dirs.append(os.path.join(base_dir, "checkpoints_reca"))
    candidate_dirs.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints_reca")))
    candidate_dirs.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RECA-paper", "Semtab", "checkpoints_new_labels")))

    # Expand candidate_dirs to also include any 1-level subdirectories inside checkpoints_reca/
    # (in case the user extracts a Google Drive .zip that creates a subfolder)
    expanded_dirs = []
    for d in candidate_dirs:
        if os.path.isdir(d):
            expanded_dirs.append(d)
            try:
                for entry in os.listdir(d):
                    sub = os.path.join(d, entry)
                    if os.path.isdir(sub):
                        expanded_dirs.append(sub)
            except Exception:
                pass

    for ckpt_dir in expanded_dirs:
        # Look for the 5 ensemble members of the target fold first (matches 5-member tolerance 1..5)
        fold_files = []
        for orig_fold in range(1, 6):
            p = os.path.join(ckpt_dir, f"semtab-RECA-finetune_dataFold_{data_fold}_origFold_{orig_fold}.pkl")
            if os.path.isfile(p):
                fold_files.append(p)
        if fold_files:
            return fold_files

        # Fallback: any .pkl checkpoint in ckpt_dir
        all_pkls = sorted([
            os.path.join(ckpt_dir, f)
            for f in os.listdir(ckpt_dir)
            if f.endswith(".pkl")
        ])
        if all_pkls:
            return all_pkls[:5]
    return []


def check_reca_available(base_dir=None):
    """Returns (is_available: bool, reason_or_device: str, num_checkpoints: int)."""
    if not HAS_TORCH:
        return False, "PyTorch / Transformers not installed", 0
    ckpts = find_reca_checkpoint_files(base_dir=base_dir)
    if not ckpts:
        return False, "No RECA checkpoints found in checkpoints_reca/ (run setup_models.py)", 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return True, device, len(ckpts)


class RECAEnsemblePredictor:
    """Manages loading and running the 5-member RECA Ensemble."""
    def __init__(self, base_dir=None, n_classes=74, data_fold=0):
        self.base_dir = base_dir
        self.n_classes = n_classes
        self.data_fold = data_fold
        self.tokenizer = None
        self.model = None
        self.state_dicts = []
        self.device = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return len(self.state_dicts) > 0
        if not HAS_TORCH:
            raise RuntimeError("PyTorch and Transformers are required to run the RECA Ensemble.")

        ckpt_paths = find_reca_checkpoint_files(self.base_dir, data_fold=self.data_fold)
        if not ckpt_paths:
            raise FileNotFoundError(
                "No RECA checkpoint files (.pkl) found in 'checkpoints_reca/'. "
                "Please run 'python setup_models.py' or switch to the Random Forest substitute classifier."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased", local_files_only=True)
        except Exception:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

        self.model = KREL(n_classes=self.n_classes).to(self.device)
        self.state_dicts = []
        for p in ckpt_paths:
            sd = torch.load(p, map_location="cpu")
            self.state_dicts.append(sd)

        self._loaded = True
        return True

    def predict_columns(self, col_texts):
        """
        Runs each column text representation through the 5 RECA ensemble members.
        Returns a list of dicts per column:
          - probs: np.ndarray of shape (n_classes,)
          - top1_idx: int
          - top1_prob: float
          - individual_preds: list of int (one per ensemble member)
          - num_different_answers: int (1..5)
        """
        self.load()
        n_cols = len(col_texts)
        encoded_inputs = []
        for text in col_texts:
            clean_text = text if (text and str(text).strip()) else "[PAD]"
            tokenized = self.tokenizer.encode_plus(
                clean_text,
                add_special_tokens=True,
                max_length=512,
                padding="max_length",
                truncation=True,
            )
            input_ids = torch.tensor([tokenized["input_ids"]], dtype=torch.long)
            encoded_inputs.append(input_ids)

        agg_logits = [None] * n_cols
        indiv_preds = [[] for _ in range(n_cols)]

        for sd in self.state_dicts:
            self.model.load_state_dict(sd)
            self.model.eval()
            with torch.no_grad():
                for idx, input_ids in enumerate(encoded_inputs):
                    ids_dev = input_ids.to(self.device)
                    out = self.model(ids_dev, ids_dev, ids_dev)
                    pred_cls = int(torch.argmax(out, dim=1).item())
                    indiv_preds[idx].append(pred_cls)
                    out_cpu = out.detach().cpu()
                    if agg_logits[idx] is None:
                        agg_logits[idx] = out_cpu
                    else:
                        agg_logits[idx] += out_cpu

        results = []
        for idx in range(n_cols):
            probs = torch.nn.functional.softmax(agg_logits[idx], dim=1).numpy()[0]
            top1_idx = int(np.argmax(probs))
            top1_prob = float(np.max(probs))
            num_diff = len(set(indiv_preds[idx]))
            results.append({
                "probs": probs,
                "top1_idx": top1_idx,
                "top1_prob": top1_prob,
                "individual_preds": indiv_preds[idx],
                "num_different_answers": num_diff,
            })
        return results

