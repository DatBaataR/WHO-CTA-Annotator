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
   - Zero-dependency Python HTTP backend (`app.py`) — launches immediately without external web server configuration.
   - Easy one-click prediction overrides, searchable dropdowns across all 73 WHO classes, top-4 model candidate probabilities, inline header editing, column removal, and full multi-step **Undo (`Ctrl + Z`) / Redo (`Ctrl + Y`)** history.

---

## Project Structure

```text
WHO-CTA-Annotator/
├── app.py                      # Web GUI server (http://127.0.0.1:8080)
├── annotate_cli.py             # Command-line batch CSV annotator
├── setup_models.py             # Downloads NLP/NER models & configures RECA checkpoints
├── requirements.txt            # Python dependencies
├── type_vocab.txt              # 74 WHO epidemiological semantic classes
├── checkpoints_rf/             # Bundled 5-fold Pruned Random Forest checkpoints (~14 MB)
├── checkpoints_reca/           # Directory for 5-fold RECA .pkl checkpoints
├── models/                     # Directory for local HF models (english_medical_ner, m2m100_418M)
├── sample_data/                # Sample WHO datasets (cholera_sample.csv, hepatitis_sample.csv)
├── src/
│   ├── rf_features.py          # Pruned 10,004-dim RF feature extractors & optimal fold params
│   ├── reca_model.py           # KREL architecture & 5-member RECA ensemble inference
│   ├── hybrid_router.py        # Hybrid routing engine, presets & soft-penalty heuristic
│   └── symptom_pipeline.py     # Coded bypass + M2M100 + BIOES NER + SpaCy + Fuzzy clustering
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
python -m spacy download en_core_web_sm
```

### 2. Optional: Download Deep Learning NLP & RECA Models

The repository works **immediately out-of-the-box** using the bundled Random Forest checkpoints in `checkpoints_rf/`. To download the HuggingFace Medical NER and M2M100 translation models (and/or copy RECA checkpoints):

```bash
python setup_models.py
```

### 3. Launch the Web GUI

```bash
python app.py
```

The browser will automatically open at `http://127.0.0.1:8080`.
- Upload any CSV file or choose one of the bundled WHO outbreak samples (`cholera_sample.csv` or `hepatitis_sample.csv`).
- Choose between **RECA + Hybrid Routing**, **Random Forest + Hybrid**, **Pure RECA**, or **Pure Random Forest**.
- Override any prediction via the column dropdown or inline text box, extract symptom columns when prompted, and press **`Ctrl + Z`** at any time to revert changes.


### 3. Launch the Web GUI
**Option A — Double-Click the Windows Executable (`.exe`)**:
Simply double-click **`WHO-CTA-Annotator.exe`** in the project folder. It automatically detects your Python / Conda environment, starts the server, and opens `http://127.0.0.1:8080` in your default browser.

**Option B — From the Terminal**:
```bash
python app.py
```

*(Optional)* If you wish to package a completely self-contained offline PyInstaller bundle (`dist/WHO-CTA-Annotator-Standalone/`) for machines without Python installed, run:
```bash
python build_exe.py
```
