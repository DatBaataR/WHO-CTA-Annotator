"""
Build a completely standalone, offline Windows executable bundle (PyInstaller)
that packages Python, the Web GUI, the Random Forest checkpoints, and dependencies
into `dist/WHO-CTA-Annotator/WHO-CTA-Annotator.exe`.

Note:
  - For GitHub distribution, the root `WHO-CTA-Annotator.exe` (10 KB native launcher)
    is already pre-built and ready to commit directly to Git.
  - Use this script only if you want to generate a self-contained multi-hundred-MB
    executable folder for distribution via USB drive or GitHub Releases.

Usage:
  python build_exe.py
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_pyinstaller():
    try:
        import PyInstaller  # noqa: F401
        return True
    except ImportError:
        print("[1/2] Installing PyInstaller into current Python environment...")
        res = subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return res.returncode == 0


def build_bundle():
    if not ensure_pyinstaller():
        print("[ERROR] Could not install PyInstaller.")
        sys.exit(1)

    sep = ";" if os.name == "nt" else ":"
    add_data = [
        f"templates{sep}templates",
        f"static{sep}static",
        f"checkpoints_rf{sep}checkpoints_rf",
        f"sample_data{sep}sample_data",
        f"type_vocab.txt{sep}.",
        f"src{sep}src",
    ]

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console",
        "--name",
        "WHO-CTA-Annotator-Standalone",
    ]
    for item in add_data:
        cmd.extend(["--add-data", item])

    hidden_imports = [
        "sklearn",
        "sklearn.ensemble",
        "sklearn.feature_extraction.text",
        "sklearn.pipeline",
        "joblib",
        "pandas",
        "numpy",
        "src.hybrid_router",
        "src.reca_model",
        "src.rf_features",
        "src.symptom_pipeline",
    ]
    for mod in hidden_imports:
        cmd.extend(["--hidden-import", mod])

    cmd.append("app.py")

    print("[2/2] Running PyInstaller standalone build...")
    res = subprocess.run(cmd, cwd=BASE_DIR)
    if res.returncode == 0:
        out_path = os.path.join(BASE_DIR, "dist", "WHO-CTA-Annotator-Standalone", "WHO-CTA-Annotator-Standalone.exe")
        print("\n" + "=" * 72)
        print("  Standalone Bundle Created Successfully!")
        print(f"  Output: {out_path}")
        print("=" * 72)
    else:
        print("[ERROR] PyInstaller build failed.")
        sys.exit(res.returncode)


if __name__ == "__main__":
    build_bundle()
