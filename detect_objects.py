#!/usr/bin/env python3
"""Run the historical-map detector over a tile grid and collect detections.

Wraps the resumable tile pipeline in the sibling ML_object_detection_production
checkout: build a status queue from a tile-grid polygon layer, download each
tile from the same WMS the model was trained on, run sliced inference, and
append the results to one GeoPackage.

The GeoPackage can be opened in QGIS while the run is in progress.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PRODUCTION_SRC = REPO_ROOT.parent / "ML_object_detection_production" / "src"

DEFAULT_TILES = "/mnt/T/mnt/trainingdata/bygningsudpegning/all_tiles_in_denmark.shp"
DEFAULT_DATATYPE = "HoejeMaalebordsblade"
DEFAULT_RESOLUTION = 0.5
DEFAULT_SLICE = 640
DEFAULT_OVERLAP = 0.0625
DEFAULT_CONFIDENCE = 0.25
# Training tiles were 0.5 m/px, so inference matches that ground resolution.
NETWORK_PREFIXES = ("/mnt/T", "/mnt/t")

log = logging.getLogger("detect_objects")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect historical map symbols across a tile grid.",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Trained YOLO .pt")
    parser.add_argument(
        "--tiles",
        default=DEFAULT_TILES,
        help="Tile-grid polygon layer (shapefile or GeoPackage)",
    )
    parser.add_argument("--tiles-layer", default=None, help="Layer name for GeoPackage input")
    parser.add_argument(
        "--id-field",
        default="kn1kmdk",
        help="Column holding the tile id (default: kn1kmdk)",
    )
    parser.add_argument(
        "--tiles-queue",
        type=Path,
        default=REPO_ROOT / "output" / "denmark_tiles.gpkg",
        help="Working status queue; reused on later runs to resume",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "denmark_detections.gpkg",
        help="Detections GeoPackage (must be on local disk)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=REPO_ROOT / "output" / "tile_images",
        help="Scratch directory for downloaded tiles",
    )
    parser.add_argument("--datatype", default=DEFAULT_DATATYPE, help="Imagery source key")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--slice-size", type=int, default=DEFAULT_SLICE)
    parser.add_argument("--overlap-ratio", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--device", default=None, help='e.g. "cuda:0" or "cpu"')
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only queue this many tiles; 0 means all of them",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Keep each tile GeoTIFF instead of deleting it after detection",
    )
    parser.add_argument(
        "--no-coverage-prescan",
        action="store_true",
        help="Try every tile instead of skipping ones the overview shows as empty",
    )
    parser.add_argument(
        "--probe-resolution",
        type=float,
        default=100.0,
        help="Ground resolution of the coverage overview in m/px",
    )
    parser.add_argument(
        "--rebuild-queue",
        action="store_true",
        help="Discard an existing queue and start the grid over",
    )
    parser.add_argument(
        "--allow-network-output",
        action="store_true",
        help="Permit a detections GeoPackage on a network mount (blocks QGIS reads)",
    )
    return parser.parse_args()


def find_token() -> str:
    matches = sorted(REPO_ROOT.glob("*token.txt"))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one file ending with token.txt in {REPO_ROOT}, "
            f"found {len(matches)}"
        )
    token = matches[0].read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(f"Token file is empty: {matches[0].name}")
    return token


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    if not PRODUCTION_SRC.is_dir():
        raise SystemExit(
            f"Sibling checkout not found: {PRODUCTION_SRC}. Clone "
            "https://github.com/SDFIdk/ML_object_detection_production next to this repo."
        )
    sys.path.insert(0, str(PRODUCTION_SRC))

    from ML_object_detection_production import tile_queue
    from ML_object_detection_production.coverage import find_uncovered_tiles
    from ML_object_detection_production.geopackage_io import ensure_detections_layer
    from ML_object_detection_production.object_detection_pipeline import run_pipeline
    from ML_object_detection_production.tiles_gpkg import (
        STATUS_DONE,
        STATUS_FAILED,
        STATUS_PENDING,
        STATUS_SKIPPED,
        count_tiles_by_status,
        create_production_tiles_gpkg,
        list_tiles_with_status,
        set_status_for_tiles,
    )

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    output = args.output.resolve()
    if not args.allow_network_output and str(output).startswith(NETWORK_PREFIXES):
        raise SystemExit(
            f"Detections GeoPackage is on a network mount: {output}\n"
            "SQLite WAL needs local disk for QGIS to read it while the run writes. "
            "Choose a local path or pass --allow-network-output."
        )

    token = find_token()
    queue_path = args.tiles_queue.resolve()
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    if args.rebuild_queue and queue_path.exists():
        queue_path.unlink()
        log.info("Removed existing queue %s", queue_path)

    if not queue_path.exists():
        log.info("Building tile queue from %s ...", args.tiles)
        create_production_tiles_gpkg(
            source_gpkg=args.tiles,
            source_layer=args.tiles_layer,
            output_gpkg=queue_path,
            id_field=args.id_field,
            all_pending=True,
            bootstrap_pending=None,
        )

        if args.limit and args.limit > 0:
            pending = list_tiles_with_status(queue_path, STATUS_PENDING)
            beyond = [t["KN1kmDK"] for t in sorted(pending, key=lambda t: t["KN1kmDK"])][
                args.limit :
            ]
            set_status_for_tiles(queue_path, beyond, STATUS_SKIPPED)
            log.info("Limit %d: left %d tile(s) pending", args.limit, args.limit)

        if not args.no_coverage_prescan:
            pending = list_tiles_with_status(queue_path, STATUS_PENDING)
            uncovered = find_uncovered_tiles(
                pending,
                args.datatype,
                token,
                probe_resolution=args.probe_resolution,
            )
            set_status_for_tiles(
                queue_path, uncovered, STATUS_SKIPPED, only_from_status=STATUS_PENDING
            )
    else:
        log.info("Resuming existing queue %s", queue_path)

    args.image_dir.mkdir(parents=True, exist_ok=True)
    ensure_detections_layer(output)

    config = {
        "tiles_geopackage": str(queue_path),
        "mode": "all",
        "image_dir": str(args.image_dir),
        "DATATYPE": args.datatype,
        "RESOLUTION": args.resolution,
        "token": token,
        "model_path": str(args.weights.resolve()),
        "geopackage_path": str(output),
        "confidence_threshold": args.confidence,
        "slice_width": args.slice_size,
        "overlap_ratio": args.overlap_ratio,
        "delete_images_after_detection": not args.keep_images,
    }
    if args.device:
        config["device"] = args.device

    log.info(
        "Queue: %d pending, %d skipped, %d done, %d failed",
        count_tiles_by_status(queue_path, STATUS_PENDING),
        count_tiles_by_status(queue_path, STATUS_SKIPPED),
        count_tiles_by_status(queue_path, STATUS_DONE),
        count_tiles_by_status(queue_path, STATUS_FAILED),
    )
    log.info("Detections: %s (safe to open in QGIS while this runs)", output)

    result = run_pipeline(config)

    log.info("=== Run summary ===")
    log.info("pipeline: %s", result)
    log.info("tiles with no map content: %d", tile_queue.blank_tile_count)
    for status in (STATUS_PENDING, STATUS_SKIPPED, STATUS_DONE, STATUS_FAILED):
        log.info("%s: %d", status, count_tiles_by_status(queue_path, status))
    log.info("Detections written to %s", output)


if __name__ == "__main__":
    main()
