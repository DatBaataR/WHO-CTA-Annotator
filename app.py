"""
WHO Epidemiological Table Annotator — Web GUI Server.
Zero-dependency HTTP/REST backend using Python's standard library (http.server)
so it runs out-of-the-box on any hardware without extra web framework dependencies.

Features:
  - Upload CSV files or load bundled WHO sample datasets (Cholera, Hepatitis)
  - Annotate with Hybrid Routing (RECA + LLM or RF Substitute + LLM) or Pure Local (RECA / RF)
  - Configurable optimal hyperparameter presets (Pareto knee-point vs Max Macro F1)
  - Automatic prompt when a 'Symptoms' column is predicted or detected
  - Interactive preview of generated 'symptom: <name>' columns with one-click remove/add custom symptom columns
  - Full state history with Undo / Revert (Ctrl + Z) and Redo (Ctrl + Y)
  - Clean, professional WHO-themed responsive interface with CSV export
"""

import os
import io
import sys
import copy
import json
import time
import uuid
import webbrowser
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import pandas as pd

if getattr(sys, "frozen", False):
    BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.hybrid_router import HybridAnnotatorEngine, HYPERPARAMETER_PRESETS
from src.symptom_pipeline import preview_symptoms_expansion, expand_symptoms_dataframe

ENGINE = HybridAnnotatorEngine(base_dir=BASE_DIR)


class SessionStore:
    """Manages active table state and full Undo/Redo snapshot history (Ctrl + Z)."""
    def __init__(self):
        self.filename = None
        self.df = None
        self.columns_meta = []
        self.annotation_info = {}
        self.undo_stack = []
        self.redo_stack = []
        self.last_symptom_preview = None

    def _snapshot(self, action_label="Edit"):
        if self.df is None:
            return None
        return {
            "action_label": action_label,
            "filename": self.filename,
            "df": self.df.copy(deep=True),
            "columns_meta": copy.deepcopy(self.columns_meta),
            "annotation_info": copy.deepcopy(self.annotation_info),
        }

    def push_undo(self, action_label="Edit"):
        snap = self._snapshot(action_label)
        if snap is not None:
            self.undo_stack.append(snap)
            if len(self.undo_stack) > 40:
                self.undo_stack.pop(0)
            self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack or self.df is None:
            return False, "Nothing to undo."
        current_snap = self._snapshot("Current")
        prev = self.undo_stack.pop()
        if current_snap is not None:
            self.redo_stack.append(current_snap)
        self.filename = prev["filename"]
        self.df = prev["df"]
        self.columns_meta = prev["columns_meta"]
        self.annotation_info = prev["annotation_info"]
        return True, f"Reverted: {prev['action_label']}"

    def redo(self):
        if not self.redo_stack or self.df is None:
            return False, "Nothing to redo."
        current_snap = self._snapshot("Redo")
        nxt = self.redo_stack.pop()
        if current_snap is not None:
            self.undo_stack.append(current_snap)
        self.filename = nxt["filename"]
        self.df = nxt["df"]
        self.columns_meta = nxt["columns_meta"]
        self.annotation_info = nxt["annotation_info"]
        return True, f"Restored: {nxt['action_label']}"

    def to_payload(self, max_preview_rows=50):
        if self.df is None:
            return {"loaded": False, "undo_count": 0, "redo_count": 0}

        preview_df = self.df.head(max_preview_rows).fillna("")
        rows = []
        for _, r in preview_df.iterrows():
            rows.append([str(val) for val in r.values])

        # Sync columns_meta with current df.columns if needed
        if len(self.columns_meta) != len(self.df.columns):
            new_meta = []
            for idx, col_name in enumerate(self.df.columns):
                new_meta.append({
                    "col_index": idx,
                    "original_header": str(col_name),
                    "current_header": str(col_name),
                    "predicted_label": "Symptom" if str(col_name).startswith("symptom:") else str(col_name),
                    "confidence": 1.0,
                    "num_different_answers": 1,
                    "triggers": [],
                    "routed_to": "Symptom Pipeline" if str(col_name).startswith("symptom:") else "User",
                    "status": "accepted" if str(col_name).startswith("symptom:") else "original",
                    "alternatives": [],
                })
            self.columns_meta = new_meta

        symptoms_candidates = []
        for c in self.columns_meta:
            cur_h = c.get("current_header", "").strip()
            pred_l = c.get("predicted_label", "").strip()
            if cur_h.lower() == "symptoms" or pred_l == "Symptoms":
                symptoms_candidates.append({
                    "col_index": c["col_index"],
                    "header": cur_h,
                    "predicted_label": pred_l,
                })

        return {
            "loaded": True,
            "filename": self.filename,
            "total_rows": len(self.df),
            "total_cols": len(self.df.columns),
            "preview_rows": rows,
            "columns_meta": self.columns_meta,
            "annotation_info": self.annotation_info,
            "symptoms_candidates": symptoms_candidates,
            "undo_count": len(self.undo_stack),
            "redo_count": len(self.redo_stack),
            "last_undo_label": self.undo_stack[-1]["action_label"] if self.undo_stack else None,
        }


SESSION = SessionStore()


def read_csv_bytes(raw_bytes):
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            text = raw_bytes.decode(enc)
            return pd.read_csv(io.StringIO(text), dtype=str)
        except Exception:
            continue
    raise ValueError("Unable to parse CSV file. Please ensure it is a valid comma-separated file.")


class AnnotatorHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet clean logging
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath, content_type):
        if not os.path.isfile(filepath):
            self.send_error(404, "File Not Found")
            return
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_file(os.path.join(BASE_DIR, "templates", "index.html"), "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            full_path = os.path.abspath(os.path.join(BASE_DIR, "static", rel))
            if not full_path.startswith(os.path.abspath(os.path.join(BASE_DIR, "static"))):
                self.send_error(403)
                return
            ext = os.path.splitext(full_path)[1].lower()
            ctype = {
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".svg": "image/svg+xml",
                ".png": "image/png",
            }.get(ext, "application/octet-stream")
            self._send_file(full_path, ctype)
        elif path == "/api/status":
            hw = ENGINE.get_hardware_status()
            samples_dir = os.path.join(BASE_DIR, "sample_data")
            samples = []
            if os.path.isdir(samples_dir):
                samples = sorted([f for f in os.listdir(samples_dir) if f.endswith(".csv")])
            hw["sample_files"] = samples
            hw["session"] = SESSION.to_payload()
            self._send_json(hw)
        elif path == "/api/export":
            if SESSION.df is None:
                self.send_error(400, "No CSV loaded")
                return
            qs = parse_qs(parsed.query)
            export_format = qs.get("format", ["renamed"])[0]
            out_df = SESSION.df.copy()

            if export_format == "renamed":
                # Use current_header for each column
                new_cols = [c["current_header"] for c in SESSION.columns_meta]
                out_df.columns = new_cols
            elif export_format == "combined":
                # Format: Original_Header (Predicted_or_Current)
                new_cols = []
                for c in SESSION.columns_meta:
                    orig = c["original_header"]
                    curr = c["current_header"]
                    if orig != curr:
                        new_cols.append(f"{orig} ({curr})")
                    else:
                        new_cols.append(curr)
                out_df.columns = new_cols

            csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
            base_name = os.path.splitext(SESSION.filename or "dataset")[0]
            dl_name = f"{base_name}_annotated.csv"
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{dl_name}"')
            self.send_header("Content-Length", str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()
        except Exception as e:
            self._send_json({"ok": False, "error": f"Invalid JSON request: {e}"}, status=400)
            return

        try:
            if path == "/api/upload":
                filename = body.get("filename", "uploaded.csv")
                csv_content = body.get("content", "")
                if not csv_content:
                    self._send_json({"ok": False, "error": "Empty CSV content."}, status=400)
                    return

                df = read_csv_bytes(csv_content.encode("utf-8"))
                if SESSION.df is not None:
                    SESSION.push_undo(f"Load file '{filename}'")
                else:
                    SESSION.undo_stack.clear()
                    SESSION.redo_stack.clear()

                SESSION.filename = filename
                SESSION.df = df
                SESSION.columns_meta = [
                    {
                        "col_index": i,
                        "original_header": str(col),
                        "current_header": str(col),
                        "predicted_label": str(col),
                        "confidence": None,
                        "num_different_answers": None,
                        "triggers": [],
                        "routed_to": "Not annotated yet",
                        "status": "original",
                        "alternatives": [],
                    }
                    for i, col in enumerate(df.columns)
                ]
                SESSION.annotation_info = {"annotated": False}

                # Auto-annotate if requested in upload payload
                if body.get("auto_annotate", True):
                    self._run_annotation_internal(body)

                self._send_json({"ok": True, "session": SESSION.to_payload()})

            elif path == "/api/load_sample":
                sample_name = body.get("sample_name", "")
                sample_path = os.path.join(BASE_DIR, "sample_data", os.path.basename(sample_name))
                if not os.path.isfile(sample_path):
                    self._send_json({"ok": False, "error": f"Sample '{sample_name}' not found."}, status=404)
                    return

                with open(sample_path, "rb") as f:
                    df = read_csv_bytes(f.read())

                if SESSION.df is not None:
                    SESSION.push_undo(f"Load sample '{sample_name}'")
                else:
                    SESSION.undo_stack.clear()
                    SESSION.redo_stack.clear()

                SESSION.filename = sample_name
                SESSION.df = df
                SESSION.columns_meta = [
                    {
                        "col_index": i,
                        "original_header": str(col),
                        "current_header": str(col),
                        "predicted_label": str(col),
                        "confidence": None,
                        "num_different_answers": None,
                        "triggers": [],
                        "routed_to": "Not annotated yet",
                        "status": "original",
                        "alternatives": [],
                    }
                    for i, col in enumerate(df.columns)
                ]
                SESSION.annotation_info = {"annotated": False}

                if body.get("auto_annotate", True):
                    self._run_annotation_internal(body)

                self._send_json({"ok": True, "session": SESSION.to_payload()})

            elif path == "/api/annotate":
                if SESSION.df is None:
                    self._send_json({"ok": False, "error": "Please upload a CSV file first."}, status=400)
                    return
                SESSION.push_undo("Run Column Type Annotation")
                self._run_annotation_internal(body)
                self._send_json({"ok": True, "session": SESSION.to_payload()})

            elif path == "/api/override":
                if SESSION.df is None:
                    self._send_json({"ok": False, "error": "No active CSV."}, status=400)
                    return
                col_idx = int(body.get("col_index", -1))
                new_header = str(body.get("new_header", "")).strip()
                new_status = str(body.get("status", "overridden"))

                if not (0 <= col_idx < len(SESSION.columns_meta)):
                    self._send_json({"ok": False, "error": "Column index out of range."}, status=400)
                    return
                if not new_header:
                    self._send_json({"ok": False, "error": "Column header cannot be empty."}, status=400)
                    return

                old_header = SESSION.columns_meta[col_idx]["current_header"]
                if old_header != new_header or SESSION.columns_meta[col_idx].get("status") != new_status:
                    SESSION.push_undo(f"Change column #{col_idx + 1} ('{old_header}' -> '{new_header}')")
                    SESSION.columns_meta[col_idx]["current_header"] = new_header
                    SESSION.columns_meta[col_idx]["status"] = new_status
                    cols = list(SESSION.df.columns)
                    cols[col_idx] = new_header
                    SESSION.df.columns = cols

                self._send_json({"ok": True, "session": SESSION.to_payload()})

            elif path == "/api/bulk_accept":
                if SESSION.df is None:
                    self._send_json({"ok": False, "error": "No active CSV."}, status=400)
                    return
                action = body.get("action", "accept_all")
                SESSION.push_undo("Accept All Predictions" if action == "accept_all" else "Reset All to Original Headers")
                cols = list(SESSION.df.columns)
                for idx, c in enumerate(SESSION.columns_meta):
                    if action == "accept_all":
                        c["current_header"] = c["predicted_label"]
                        c["status"] = "accepted"
                    else:
                        c["current_header"] = c["original_header"]
                        c["status"] = "original"
                    cols[idx] = c["current_header"]
                SESSION.df.columns = cols
                self._send_json({"ok": True, "session": SESSION.to_payload()})

            elif path == "/api/delete_column":
                if SESSION.df is None:
                    self._send_json({"ok": False, "error": "No active CSV."}, status=400)
                    return
                col_idx = int(body.get("col_index", -1))
                if not (0 <= col_idx < len(SESSION.columns_meta)):
                    self._send_json({"ok": False, "error": "Invalid column index."}, status=400)
                    return
                col_title = SESSION.columns_meta[col_idx]["current_header"]
                SESSION.push_undo(f"Remove column '{col_title}'")
                # Drop positional column
                keep_positions = [i for i in range(len(SESSION.df.columns)) if i != col_idx]
                SESSION.df = SESSION.df.iloc[:, keep_positions].copy()
                SESSION.columns_meta.pop(col_idx)
                for i, c in enumerate(SESSION.columns_meta):
                    c["col_index"] = i
                self._send_json({"ok": True, "session": SESSION.to_payload()})

            elif path == "/api/add_custom_symptom_column":
                if SESSION.df is None:
                    self._send_json({"ok": False, "error": "No active CSV."}, status=400)
                    return
                symptom_name = str(body.get("symptom_name", "")).strip()
                default_val = bool(body.get("default_value", False))
                source_col_idx = body.get("source_col_index", None)
                if not symptom_name:
                    self._send_json({"ok": False, "error": "Please enter a symptom name."}, status=400)
                    return

                col_title = symptom_name if symptom_name.lower().startswith("symptom:") else f"symptom: {symptom_name.lower()}"
                keyword = col_title.split(":", 1)[1].strip().lower()

                SESSION.push_undo(f"Add symptom column '{col_title}'")

                if source_col_idx is not None and 0 <= int(source_col_idx) < len(SESSION.df.columns):
                    src_series = SESSION.df.iloc[:, int(source_col_idx)]
                    vals = [
                        ("True" if (not pd.isna(v) and keyword in str(v).lower()) else "False")
                        for v in src_series
                    ]
                else:
                    vals = ["True" if default_val else "False"] * len(SESSION.df)

                SESSION.df[col_title] = vals
                new_idx = len(SESSION.columns_meta)
                SESSION.columns_meta.append({
                    "col_index": new_idx,
                    "original_header": col_title,
                    "current_header": col_title,
                    "predicted_label": "Symptom",
                    "confidence": 1.0,
                    "num_different_answers": 1,
                    "triggers": [],
                    "routed_to": "Custom Symptom Added",
                    "status": "accepted",
                    "alternatives": [{"label": "Symptom", "prob": 1.0}, {"label": "Boolean", "prob": 0.0}],
                })
                self._send_json({"ok": True, "session": SESSION.to_payload()})

            elif path == "/api/symptoms/preview":
                if SESSION.df is None:
                    self._send_json({"ok": False, "error": "No active CSV."}, status=400)
                    return
                col_idx = body.get("col_index", None)
                target_col = None
                if col_idx is not None and 0 <= int(col_idx) < len(SESSION.df.columns):
                    target_col = SESSION.df.columns[int(col_idx)]
                else:
                    # Find first column predicted as Symptoms or named Symptoms
                    for idx, c in enumerate(SESSION.columns_meta):
                        if c["current_header"].lower() == "symptoms" or c["predicted_label"] == "Symptoms":
                            target_col = SESSION.df.columns[idx]
                            col_idx = idx
                            break

                if target_col is None:
                    self._send_json({"ok": False, "error": "No 'Symptoms' column found to extract."}, status=400)
                    return

                preview = preview_symptoms_expansion(SESSION.df, target_col=target_col, base_dir=BASE_DIR)
                SESSION.last_symptom_preview = {
                    "col_index": int(col_idx),
                    "target_col": target_col,
                    "detected_language": preview["detected_language"],
                    "columns": preview["columns"],
                    "counts": preview["counts"],
                }

                # Prepare sample rows preview for the modal
                generated_list = [
                    {"name": col_name, "true_count": count, "total_rows": len(SESSION.df)}
                    for col_name, count in preview["counts"].items()
                ]
                self._send_json({
                    "ok": True,
                    "col_index": int(col_idx),
                    "target_col": target_col,
                    "detected_language": preview["detected_language"],
                    "generated_columns": generated_list,
                })

            elif path == "/api/symptoms/apply":
                if SESSION.df is None:
                    self._send_json({"ok": False, "error": "No active CSV."}, status=400)
                    return
                col_idx = int(body.get("col_index", -1))
                selected_cols = body.get("selected_columns", [])
                custom_cols = body.get("custom_columns", [])
                drop_original = bool(body.get("drop_original", True))

                if not (0 <= col_idx < len(SESSION.df.columns)):
                    self._send_json({"ok": False, "error": "Invalid Symptoms column index."}, status=400)
                    return

                target_col = SESSION.df.columns[col_idx]
                SESSION.push_undo(f"Expand Symptoms column '{target_col}'")

                expanded_df = expand_symptoms_dataframe(
                    SESSION.df,
                    target_col=target_col,
                    selected_symptom_cols=selected_cols,
                    custom_symptom_cols=custom_cols,
                    drop_original=drop_original,
                    base_dir=BASE_DIR,
                )

                # Update metadata for remaining and newly added symptom columns
                new_meta = []
                old_by_name = {c["current_header"]: c for c in SESSION.columns_meta}
                for idx, col_name in enumerate(expanded_df.columns):
                    if col_name in old_by_name and not str(col_name).startswith("symptom:"):
                        entry = copy.deepcopy(old_by_name[col_name])
                        entry["col_index"] = idx
                        new_meta.append(entry)
                    else:
                        # Convert boolean values to string "True"/"False" for consistent CSV display
                        expanded_df[col_name] = expanded_df[col_name].astype(str)
                        new_meta.append({
                            "col_index": idx,
                            "original_header": str(col_name),
                            "current_header": str(col_name),
                            "predicted_label": "Symptom",
                            "confidence": 1.0,
                            "num_different_answers": 1,
                            "triggers": [],
                            "routed_to": "Symptom Extraction Pipeline",
                            "status": "accepted",
                            "alternatives": [{"label": "Symptom", "prob": 1.0}, {"label": "Boolean", "prob": 0.0}],
                        })

                SESSION.df = expanded_df
                SESSION.columns_meta = new_meta
                self._send_json({"ok": True, "session": SESSION.to_payload()})

            elif path == "/api/undo":
                ok, msg = SESSION.undo()
                self._send_json({"ok": ok, "message": msg, "session": SESSION.to_payload()})

            elif path == "/api/redo":
                ok, msg = SESSION.redo()
                self._send_json({"ok": ok, "message": msg, "session": SESSION.to_payload()})

            else:
                self._send_json({"ok": False, "error": f"Unknown endpoint {path}"}, status=404)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _run_annotation_internal(self, params):
        mode = params.get("mode", "reca_pareto")
        conf_threshold = params.get("conf_threshold", None)
        ensemble_tolerance = params.get("ensemble_tolerance", None)
        use_weak_classes = params.get("use_weak_classes", None)
        use_soft_penalty = params.get("use_soft_penalty", None)
        api_key = params.get("api_key", None)
        auto_apply = params.get("auto_apply_predictions", False)

        # Temporarily restore original headers on a copy so re-running inference uses original column headers
        eval_df = SESSION.df.copy()
        if len(SESSION.columns_meta) == len(eval_df.columns):
            eval_df.columns = [c["original_header"] for c in SESSION.columns_meta]

        res = ENGINE.annotate_dataframe(
            eval_df,
            mode=mode,
            conf_threshold=conf_threshold,
            ensemble_tolerance=ensemble_tolerance,
            use_weak_classes=use_weak_classes,
            use_soft_penalty=use_soft_penalty,
            api_key=api_key,
        )

        new_meta = []
        new_df_cols = list(SESSION.df.columns)
        for idx, col_rep in enumerate(res["columns"]):
            orig_h = SESSION.columns_meta[idx]["original_header"] if idx < len(SESSION.columns_meta) else col_rep["original_header"]
            pred_lbl = col_rep["predicted_label"]
            curr_h = pred_lbl if auto_apply else (SESSION.columns_meta[idx]["current_header"] if idx < len(SESSION.columns_meta) else orig_h)
            status = "accepted" if (auto_apply or curr_h == pred_lbl) else "recommended"
            new_df_cols[idx] = curr_h

            new_meta.append({
                "col_index": idx,
                "original_header": orig_h,
                "current_header": curr_h,
                "predicted_label": pred_lbl,
                "local_candidate": col_rep["local_candidate"],
                "confidence": col_rep["confidence"],
                "num_different_answers": col_rep["num_different_answers"],
                "triggers": col_rep["triggers"],
                "needs_escalation": col_rep["needs_escalation"],
                "routed_to": col_rep["routed_to"],
                "status": status,
                "alternatives": col_rep["alternatives"],
            })

        SESSION.df.columns = new_df_cols
        SESSION.columns_meta = new_meta
        SESSION.annotation_info = {
            "annotated": True,
            "mode": mode,
            "backbone_used": res["backbone_used"],
            "hybrid_enabled": res["hybrid_enabled"],
            "conf_threshold": res["conf_threshold"],
            "ensemble_tolerance": res["ensemble_tolerance"],
            "use_weak_classes": res["use_weak_classes"],
            "use_soft_penalty": res["use_soft_penalty"],
            "fallback_notice": res["fallback_notice"],
            "routed_llm_count": res["routed_llm_count"],
            "routed_local_count": res["routed_local_count"],
            "detected_symptoms_columns": res["detected_symptoms_columns"],
        }


def run_server(host="127.0.0.1", port=8080, open_browser=True):
    server = HTTPServer((host, port), AnnotatorHTTPRequestHandler)
    url = f"http://{host}:{port}"
    print("=" * 75)
    print("  WHO Epidemiological Column Type Annotator & Symptom Extraction Suite")
    print("=" * 75)
    print(f"  Web GUI running at : {url}")
    print("  Undo / Revert      : Ctrl + Z (or Revert button in top bar)")
    print("  Stop Server        : Ctrl + C")
    print("=" * 75)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the WHO CTA Web Annotator GUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port number (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open web browser automatically")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)

