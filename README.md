# WHO Epidemiological Column Type Annotator & Symptom Extraction Suite

An end-to-end semantic column type annotation (CTA) and multilingual clinical symptom extraction tool developed for World Health Organization (WHO) epidemiological outbreak line-lists and surveillance tables.

## Key Features

1. **RECA Ensemble with Optimal Hybrid Routing**
   - Uses a 5-member fine-tuned multilingual BERT (`bert-base-multilingual-uncased`) **KREL (RECA)** ensemble over 74 epidemiological semantic classes, enhanced with a table-level greedy soft-penalty heuristic (`default_penalty = 0.05`, `date_penalty = 0.10`).
   - Escalates ambiguous columns to an **ArcheType-LLM** router based on three triggers:
     1. **Ensemble Disagreement**: Number of distinct answers across the 5 members $N_{\text{diff}} > \tau_{\text{ens}}$.
     2. **Low Confidence**: Maximum softmax probability $p_{\max} < \theta_{\text{conf}}$.
     3. **Weak Domain Classes**: 23 rare/hard epidemiological concepts (`Pathogen`, `Travel_history`, `Date_onset`, etc.).
   - Includes both **Pareto-Optimal** ($\theta_{\text{conf}} = 0.50, \tau_{\text{ens}} = 2$) and **Max Macro F1** ($\theta_{\text{conf}} = 0.50, \tau_{\text{ens}} = 1$) presets found during cross-validation testing.

2. **Hardware-Friendly Random Forest Substitute**
   - For systems without a dedicated GPU or where full BERT ensembles cannot be run, the project includes a pre-trained **Pruned Random Forest Classifier** (10,004 features: Word TF-IDF + Character 2–4g TF-IDF + Character Ratio features; excluding noisy numerical/date summary statistics).
   - Partitions trees into 5 disjoint sub-forests to support the exact same hybrid routing architecture using its calibrated **Pareto-Optimal** ($\theta_{\text{conf}} = 0.30, \tau_{\text{ens}} = 2$) and **Max Macro F1** ($\theta_{\text{conf}} = 0.55, \tau_{\text{ens}} = 1$) presets, or runs 100% locally in milliseconds (`pure_rf`).

3. **Multilingual Symptom Extraction Pipeline**
   - Automatically detects when a column is named or predicted as `Symptoms` and prompts the user in the Web GUI.
   - Combines a **Coded-Data Bypass** (for semicolon/pipe/underscore WHO & TESSy clinical codes), **M2M100-418M** multilingual translation, **HUMADEX/english_medical_ner** with a custom **BIOES full-word span merger**, **SpaCy lemmatization & severity filtering**, and **Fuzzy Clustering (>85% similarity)** to expand messy symptom notes into clean boolean `symptom: <name>` columns.
   - Interactive modal previews all generated `symptom: <name>` columns with positive case counts, allowing effortless one-click removal or addition of custom symptom columns.

4. **Clean, Professional Web GUI with Undo / Revert (`Ctrl + Z`)**
   - Zero-dependency Python HTTP backend (`app.py`) and one-click Windows executable launcher (`WHO-CTA-Annotator.exe`).
   - Easy one-click prediction overrides, searchable dropdowns across all 73 WHO classes, top-4 model candidate probabilities, inline header editing, column removal, and full multi-step **Undo (`Ctrl + Z`) / Redo (`Ctrl + Y`)** history.

---

## Project Structure

```text
WHO-CTA-Annotator/
├── WHO-CTA-Annotator.exe       # One-click Windows launcher for the Web GUI
├── app.py                      # Web GUI server (http://127.0.0.1:8080)
├── annotate_cli.py             # Command-line batch CSV annotator
├── setup_models.py             # Downloads NLP/NER models & configures RECA checkpoints
├── build_exe.py                # Optional PyInstaller standalone offline bundle builder
├── requirements.txt            # Python dependencies
├── type_vocab.txt              # 74 WHO epidemiological semantic classes
├── checkpoints_rf/             # Bundled 5-fold Pruned Random Forest checkpoints (~14.3 MB)
├── checkpoints_reca/           # Place Google Drive RECA .pkl checkpoints here
├── models/                     # Directory for local HF models (english_medical_ner, m2m100_418M)
├── sample_data/                # Sample WHO datasets (cholera_sample.csv, hepatitis_sample.csv)
├── src/
│   ├── rf_features.py          # Pruned 10,004-dim RF feature extractors & optimal fold params
│   ├── reca_model.py           # KREL architecture & 5-member RECA ensemble inference
│   ├── hybrid_router.py        # Hybrid routing engine, presets & soft-penalty heuristic
│   ├── symptom_pipeline.py     # Coded bypass + M2M100 + BIOES NER + SpaCy + Fuzzy clustering
│   └── launcher.cs             # C# source for WHO-CTA-Annotator.exe
├── templates/
│   └── index.html              # Web GUI frontend layout
└── static/
    ├── style.css               # Clean WHO-themed styling
    └── app.js                  # Interactive frontend controller & Ctrl+Z history
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download RECA Checkpoints (Google Drive) & NLP Models

The repository works **immediately out-of-the-box** using the bundled Random Forest checkpoints in `checkpoints_rf/`.

#### A. RECA Ensemble Checkpoints (`checkpoints_reca/`)
Because the fine-tuned RECA (`KREL`) checkpoints are **~639 MB each** (~3.2 GB total for the 5-member ensemble), they are hosted on **Google Drive**:
1. Download the 5 ensemble checkpoint files from Google Drive (`[Insert Your Google Drive Link Here]`):
   - `semtab-RECA-finetune_dataFold_0_origFold_1.pkl`
   - `semtab-RECA-finetune_dataFold_0_origFold_2.pkl`
   - `semtab-RECA-finetune_dataFold_0_origFold_3.pkl`
   - `semtab-RECA-finetune_dataFold_0_origFold_4.pkl`
   - `semtab-RECA-finetune_dataFold_0_origFold_5.pkl`
2. Place them inside the **`checkpoints_reca/`** folder (or download them automatically via `python setup_models.py --gdrive-url "<YOUR_GOOGLE_DRIVE_LINK>"`).

#### B. Symptom Extraction & Base BERT Models
Run `setup_models.py` to download the HuggingFace Medical NER (`HUMADEX/english_medical_ner`), multilingual translation (`facebook/m2m100_418M`), SpaCy dictionary (`en_core_web_sm`), and `bert-base-multilingual-uncased`:

```bash
python setup_models.py
```

### 3. Launch the Web GUI

**Option A — Double-Click the Windows Executable (`.exe`)**:
Simply double-click **`WHO-CTA-Annotator.exe`** in the project folder. It automatically locates your Python / Conda environment, starts the server, and opens `http://127.0.0.1:8080` in your default browser.

**Option B — From the Terminal**:
```bash
python app.py
```

- Upload any CSV file or choose one of the bundled WHO outbreak samples (`cholera_sample.csv` or `hepatitis_sample.csv`).
- Choose between **RECA + Hybrid Routing**, **Random Forest + Hybrid**, **Pure RECA**, or **Pure Random Forest**.
- Override any prediction via the column dropdown or inline text box, extract symptom columns when prompted, and press **`Ctrl + Z`** at any time to revert changes.

---

## 4. Continual Fine-Tuning on New WHO Data (`finetune.py`)

WHO employees can continually fine-tune the 5-member **RECA Ensemble** (and/or retrain the **Random Forest Substitute**) on newly collected outbreak tables directly using the existing fine-tuned WHO checkpoints in `checkpoints_reca/` — **the original 275-class SemTab pre-trained checkpoints are NOT required**.

> **Why continual fine-tuning on the existing WHO checkpoints is recommended:**
> The original SemTab checkpoints were trained on 275 general DBpedia classes, meaning their classification layers (`fcc_tar`, `fcc_rel`, `fcc_sub`) had to be discarded and randomly initialized when adapting to the 74 WHO classes. By starting from the 5 WHO-finetuned RECA checkpoints (`semtab-RECA-finetune_dataFold_0_origFold_1..5.pkl`) with a gentle learning rate (`--lr 5e-6`), the model preserves both its domain-adapted multilingual BERT representations and its warm-started 74-class epidemiological classification head.

### Supported Training Data Formats (`--train-data`)

`finetune.py` accepts **two convenient formats** (see examples in `sample_data/finetune_example/`):

#### Format A — Annotated WHO Table CSV(s) (Recommended / Zero-Prep)
Simply export your annotated table(s) from the **WHO-CTA-Annotator Web GUI** (either with **Renamed Headers** like `ID`, `Age`, `Date_onset`, `Symptoms`, or **Combined Headers** like `Patient_ID (ID)`, `Onset_Date (Date_onset)`) and place one or more `.csv` files in a folder:

| `Patient_ID (ID)` | `Patient_Age (Age)` | `Sex (Gender)` | `Onset_Date (Date_onset)` | `Clinical_Notes (Symptoms)` | `Vital_Status (Outcome)` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| WHO-2026-001 | 34 | Female | 2026-03-01 | watery diarrhea, vomiting | Alive |
| WHO-2026-002 | 12 | Male | 2026-03-02 | severe dehydration | Alive |

`finetune.py` automatically serializes each column into `<OriginalHeader> : <val_1> <val_2> ...` and maps the target class name to its ID (`0..73`) in `type_vocab.txt`.

#### Format B — Pre-Formatted Column-Level CSV
A single `.csv` file where each row is one serialized column with `label` (class name from `type_vocab.txt` or integer ID `0..73`) and `data` (`<OriginalHeader> : <cell values>`):

```csv
table_id,column_index,label,data
who_outbreak_1,0,ID,Patient_ID : WHO-2026-001 WHO-2026-002 WHO-2026-003
who_outbreak_1,1,Age,Patient_Age : 34 12 67 45 29 51 8 42
who_outbreak_1,2,Gender,Sex : Female Male Female Male
who_outbreak_1,3,Date_onset,Onset_Date : 2026-03-01 2026-03-02 2026-03-04
```

### Running `finetune.py`

```bash
# 1. Continually fine-tune the 5 RECA Ensemble checkpoints on a folder of annotated CSV tables:
python finetune.py --train-data ./sample_data/finetune_example --model reca --epochs 8 --lr 5e-6

# 2. Fine-tune and overwrite checkpoints_reca/ and checkpoints_rf/ in-place for immediate use in the GUI:
python finetune.py --train-data ./sample_data/finetune_example --model both --in-place
```
