#!/usr/bin/env python3
"""Run the README workflow: WMS dataset, train, SAHI inference, GeoPackage export."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SIBLING_REPO = REPO_ROOT.parent / "ML_object_detection"
SIBLING_TRAIN = SIBLING_REPO / "src/ML_object_detection/train.py"
SIBLING_INFER = SIBLING_REPO / "src/ML_object_detection/infer_with_sahi.py"
PREPARE = REPO_ROOT / "src/danish-historical-map-yolo/prepare_wms_yolo_dataset.py"
EXPORT = REPO_ROOT / "src/danish-historical-map-yolo/export_prediction_candidates_to_gpkg.py"

DATASET_DIR = REPO_ROOT / "data/generated/hoje_maalebordsblade"
DATASET_YAML = DATASET_DIR / "dataset.yaml"
VAL_IMAGES = DATASET_DIR / "images/val"
MANIFEST = DATASET_DIR / "tile_manifest.csv"
INFER_OUT = REPO_ROOT / "output/labelme"
GPKG_OUT = REPO_ROOT / "output/prediction_review.gpkg"
TRAIN_RUN_NAME = "hoje_maalebordsblade_experiment"

MIN_GPU_FREE_MIB = 6144


def find_token_file() -> Path:
    matches = sorted(REPO_ROOT.glob("*token.txt"))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one file ending with token.txt in {REPO_ROOT}, found {len(matches)}"
        )
    return matches[0]


def python_command() -> list[str]:
    if shutil.which("mamba"):
        return ["mamba", "run", "-n", "ML_object_detection", "python"]
    if shutil.which("conda"):
        return ["conda", "run", "-n", "ML_object_detection", "python"]
    return [sys.executable]


def gpu_free_mib() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return max(values) if values else None


def choose_device() -> str:
    free = gpu_free_mib()
    if free is not None and free >= MIN_GPU_FREE_MIB:
        print(f"Using cuda:0 ({free} MiB GPU memory free)")
        return "cuda:0"
    if free is not None:
        print(f"Using cpu ({free} MiB GPU memory free, need {MIN_GPU_FREE_MIB})")
    else:
        print("Using cpu (no usable nvidia-smi GPU info)")
    return "cpu"


def ensure_sibling_repo() -> None:
    if SIBLING_REPO.is_dir() and SIBLING_TRAIN.is_file() and SIBLING_INFER.is_file():
        return
    if SIBLING_REPO.is_dir():
        raise SystemExit(
            f"Found {SIBLING_REPO} but missing train.py or infer_with_sahi.py under src/ML_object_detection/"
        )
    print(f"Cloning ML_object_detection into {SIBLING_REPO.parent}")
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/SDFIdk/ML_object_detection.git",
            str(SIBLING_REPO),
        ],
        check=True,
    )


def run_step(label: str, cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def run_step_checked(label: str, cmd: list[str], env: dict[str, str]) -> str:
    result = run_step(label, cmd, env)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        raise SystemExit(f"{label} failed with exit code {result.returncode}")
    return result.stdout


def parse_best_weights(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("BEST_WEIGHTS="):
            path = Path(line.split("=", 1)[1].strip())
            if path.is_file():
                return path
    matches = sorted((REPO_ROOT / "runs/detect").glob(f"{TRAIN_RUN_NAME}*/weights/best.pt"))
    if matches:
        return matches[-1].resolve()
    raise SystemExit("Could not find best.pt after training (no BEST_WEIGHTS= in output)")


def cuda_oom(stderr: str) -> bool:
    return bool(re.search(r"CUDA out of memory|OutOfMemoryError", stderr, re.I))


def main() -> None:
    ensure_sibling_repo()
    token_path = find_token_file()
    api_key = token_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit(f"Token file is empty: {token_path.name}")

    env = os.environ.copy()
    env["DATAFORDELER_APIKEY"] = api_key
    py = python_command()

    prepare_cmd = py + [
        str(PREPARE),
        "--gpkg",
        str(REPO_ROOT / "data/annotations/hoje_maalebordsblade_annotations.gpkg"),
        "--areas-layer",
        "hoje_maalebordsblad_annoteringsomraade",
        "--boxes-layer",
        "bbox_alle_objekter_v3",
        "--class-field",
        "layer",
        "--classes",
        "engtotter",
        "mosepolygoner",
        "vandlinjer",
        "siv",
        "lyng",
        "--output",
        str(DATASET_DIR),
        "--target-crs",
        "EPSG:25832",
        "--pixel-size",
        "0.5",
        "--tile-size",
        "640",
        "--overlap",
        "80",
        "--val-fraction",
        "0.2",
        "--seed",
        "4",
        "--min-area-coverage",
        "1.0",
        "--mask-outside-areas",
        "--include-edge-object-tiles",
        "--edge-box-padding-pixels",
        "10",
    ]
    run_step_checked("Prepare WMS YOLO dataset", prepare_cmd, env)
    if not DATASET_YAML.is_file():
        raise SystemExit(f"Missing {DATASET_YAML}")

    device = choose_device()
    train_cmd = py + [
        str(SIBLING_TRAIN),
        "--data",
        str(DATASET_YAML),
        "--weights",
        "yolov8n.pt",
        "--epochs",
        "50",
        "--imgsz",
        "640",
        "--device",
        device,
        "--name",
        TRAIN_RUN_NAME,
    ]
    train_result = run_step("Train YOLO", train_cmd, env)
    if train_result.stdout:
        print(train_result.stdout, end="" if train_result.stdout.endswith("\n") else "\n")
    if train_result.returncode != 0:
        if device.startswith("cuda") and cuda_oom(train_result.stderr or train_result.stdout):
            print("Training hit GPU OOM; retrying on cpu", file=sys.stderr)
            train_cmd_cpu = train_cmd.copy()
            device_idx = train_cmd_cpu.index("--device") + 1
            train_cmd_cpu[device_idx] = "cpu"
            train_stdout = run_step_checked("Train YOLO (cpu retry)", train_cmd_cpu, env)
        else:
            if train_result.stderr:
                print(train_result.stderr, file=sys.stderr)
            raise SystemExit(f"Train failed with exit code {train_result.returncode}")
    else:
        train_stdout = train_result.stdout

    best_weights = parse_best_weights(train_stdout)
    print(f"Using weights: {best_weights}")

    infer_cmd = py + [
        str(SIBLING_INFER),
        "--weights",
        str(best_weights),
        "--folder_with_images",
        str(VAL_IMAGES),
        "--result_folder",
        str(INFER_OUT),
        "--slice_width",
        "640",
        "--overlap_ratio",
        "0.0625",
    ]
    run_step_checked("SAHI inference", infer_cmd, env)
    json_files = list(INFER_OUT.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No LabelMe JSON files in {INFER_OUT}")

    export_cmd = py + [
        str(EXPORT),
        "--weights",
        str(best_weights),
        "--images",
        str(VAL_IMAGES),
        "--manifest",
        str(MANIFEST),
        "--annotations-gpkg",
        str(REPO_ROOT / "data/annotations/hoje_maalebordsblade_annotations.gpkg"),
        "--annotations-layer",
        "bbox_alle_objekter_v3",
        "--class-field",
        "layer",
        "--output",
        str(GPKG_OUT),
        "--confidence",
        "0.25",
        "--match-iou",
        "0.20",
        "--dedupe-iou",
        "0.40",
        "--device",
        "cpu",
        "--overwrite",
    ]
    run_step_checked("Export prediction candidates", export_cmd, env)
    if not GPKG_OUT.is_file():
        raise SystemExit(f"Missing {GPKG_OUT}")

    print("\nAll pipeline steps completed successfully.")


if __name__ == "__main__":
    main()
