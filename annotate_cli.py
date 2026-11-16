"""
annotate_cli.py — Command-Line Batch Annotation & Symptom Extraction Tool for WHO Datasets.

Examples:
  # Annotate with Pareto-optimal RECA Hybrid routing:
  python annotate_cli.py --input sample_data/cholera_sample.csv --output cholera_annotated.csv --mode reca_pareto

  # Annotate on lightweight hardware using Pruned Random Forest + Hybrid routing and split Symptoms:
  python annotate_cli.py --input sample_data/hepatitis_sample.csv --output hepatitis_expanded.csv --mode rf_pareto --expand-symptoms
"""

import os
import sys
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.hybrid_router import HybridAnnotatorEngine, HYPERPARAMETER_PRESETS
from src.symptom_pipeline import expand_symptoms_dataframe


def main():
    parser = argparse.ArgumentParser(description="Annotate epidemiological CSV tables via Hybrid RECA / RF + Symptom Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output annotated CSV file")
    parser.add_argument(
        "--mode",
        type=str,
        default="reca_pareto",
        choices=list(HYPERPARAMETER_PRESETS.keys()),
        help="Routing preset (default: reca_pareto)",
    )
    parser.add_argument("--conf-threshold", type=float, default=None, help="Override confidence threshold theta_conf")
    parser.add_argument("--ensemble-tolerance", type=int, default=None, help="Override ensemble tolerance tau_ens (1..5)")
    parser.add_argument("--expand-symptoms", action="store_true", help="Automatically expand any column predicted as 'Symptoms' into boolean symptom columns")
    parser.add_argument("--api-key", type=str, default=None, help="Optional Google Gemini API key for live LLM escalation")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str)
    engine = HybridAnnotatorEngine(base_dir=BASE_DIR)

    print(f"Running annotation on '{args.input}' ({len(df)} rows, {len(df.columns)} columns) with mode='{args.mode}'...")
    res = engine.annotate_dataframe(
        df,
        mode=args.mode,
        conf_threshold=args.conf_threshold,
        ensemble_tolerance=args.ensemble_tolerance,
        api_key=args.api_key,
    )

    if res["fallback_notice"]:
        print(f"[Notice] {res['fallback_notice']}")

    print(f"\n{'#':<4} | {'Original Header':<32} | {'Predicted Class':<26} | {'Conf':<6} | {'Route'}")
    print("-" * 95)
    new_headers = []
    for c in res["columns"]:
        print(
            f"{c['col_index']+1:<4} | {c['original_header'][:30]:<32} | "
            f"{c['predicted_label']:<26} | {c['confidence']*100:>4.0f}%  | {c['routed_to']}"
        )
        new_headers.append(c["predicted_label"])

    out_df = df.copy()
    out_df.columns = new_headers

    if args.expand_symptoms and "Symptoms" in out_df.columns:
        print("\n[Symptom Pipeline] Expanding 'Symptoms' column into boolean symptom columns...")
        out_df = expand_symptoms_dataframe(out_df, target_col="Symptoms", base_dir=BASE_DIR)
        sym_cols = [col for col in out_df.columns if str(col).startswith("symptom:")]
        print(f"[Symptom Pipeline] Generated {len(sym_cols)} boolean symptom columns: {sym_cols}")

    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nSaved annotated CSV to: {args.output}")


if __name__ == "__main__":
    main()
