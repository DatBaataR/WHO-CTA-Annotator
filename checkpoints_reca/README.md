# RECA Ensemble Checkpoints (`checkpoints_reca/`)

Because each fine-tuned multilingual BERT (`KREL`) checkpoint is **~639 MB** (~3.2 GB total for the 5-member ensemble), they are hosted on **Google Drive** rather than tracked directly in Git.

## Required Files (5-Member Ensemble)
Download the following 5 checkpoint files from the Google Drive folder and place them directly inside this `checkpoints_reca/` directory:

1. `semtab-RECA-finetune_dataFold_0_origFold_1.pkl` (~639 MB)
2. `semtab-RECA-finetune_dataFold_0_origFold_2.pkl` (~639 MB)
3. `semtab-RECA-finetune_dataFold_0_origFold_3.pkl` (~639 MB)
4. `semtab-RECA-finetune_dataFold_0_origFold_4.pkl` (~639 MB)
5. `semtab-RECA-finetune_dataFold_0_origFold_5.pkl` (~639 MB)

> **Note:** If Google Drive downloads the folder as a `.zip` archive, simply extract the `.pkl` files into `checkpoints_reca/` (even if extracting creates a subfolder inside `checkpoints_reca/`, the application will automatically detect the `.pkl` files).

## Automated Download via `setup_models.py`
You can also download the checkpoints from Google Drive via the setup script:
```bash
python setup_models.py --gdrive-url "<YOUR_GOOGLE_DRIVE_LINK>"
```

## Hardware Fallback
If this folder is left empty, the Web GUI and CLI will automatically fall back to the bundled **5-Fold Pruned Random Forest Substitute** (`checkpoints_rf/`), which runs out-of-the-box on any CPU.
