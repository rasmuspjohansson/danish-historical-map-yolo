"""Export YOLO detections without matching GIS annotations to a GeoPackage.

The script runs a trained Ultralytics YOLO model on georeferenced dataset tiles.
Tile bounds are read from tile_manifest.csv, predictions are transformed to
EPSG:25832, duplicate detections from overlapping tiles are removed, and each
prediction is compared with the original annotation layer. Unmatched detections
are written as review candidates for QGIS.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box as make_box
from ultralytics import YOLO


DEFAULT_TARGET_CRS = "EPSG:25832"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unmatched YOLO predictions to a QGIS GeoPackage."
    )
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--target-crs",
        help=(
            "CRS of manifest bounds. Defaults to the manifest 'crs' column, "
            f"or {DEFAULT_TARGET_CRS} for older manifests."
        ),
    )
    parser.add_argument("--annotations-gpkg", required=True, type=Path)
    parser.add_argument("--annotations-layer", default="bbox_alle_objekter")
    parser.add_argument("--class-field", default="layer")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.20,
        help="Minimum overlap with any annotation for a prediction to count as matched",
    )
    parser.add_argument(
        "--dedupe-iou",
        type=float,
        default=0.40,
        help="Overlap threshold for removing duplicate predictions from overlapping tiles",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def intersection_over_union(first, second) -> float:
    intersection = first.intersection(second).area
    if intersection <= 0:
        return 0.0
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filename", "minx", "miny", "maxx", "maxy"}
    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError("Manifest is missing columns: " + ", ".join(sorted(missing)))
    return {row["filename"]: row for row in rows}


def class_name(model: YOLO, class_id: int) -> str:
    names = model.names
    if isinstance(names, dict):
        return str(names[class_id])
    return str(names[class_id])


def remove_spatial_duplicates(records: list[dict], threshold: float) -> list[dict]:
    """Class-aware greedy non-maximum suppression in map coordinates."""
    kept: list[dict] = []
    for record in sorted(records, key=lambda item: item["confidence"], reverse=True):
        duplicate = False
        for existing in kept:
            if record["class_id"] != existing["class_id"]:
                continue
            if intersection_over_union(record["geometry"], existing["geometry"]) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(record)
    return kept


def main() -> None:
    args = parse_args()
    for path, description in (
        (args.weights, "weights"),
        (args.images, "image directory"),
        (args.manifest, "tile manifest"),
        (args.annotations_gpkg, "annotation GeoPackage"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Could not find {description}: {path}")
    for name, value in (("--confidence", args.confidence), ("--match-iou", args.match_iou), ("--dedupe-iou", args.dedupe_iou)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --overwrite to replace it."
        )

    manifest = read_manifest(args.manifest)
    annotations = gpd.read_file(args.annotations_gpkg, layer=args.annotations_layer)
    if annotations.crs is None:
        raise ValueError("The annotation layer has no CRS")
    if args.class_field not in annotations.columns:
        raise KeyError(f"Field '{args.class_field}' is missing from the annotation layer")
    manifest_crs_values = {
        row.get("crs", "").strip()
        for row in manifest.values()
        if row.get("crs", "").strip()
    }
    if len(manifest_crs_values) > 1:
        raise ValueError("Manifest contains more than one CRS")
    manifest_crs = next(iter(manifest_crs_values), "")
    target_crs = args.target_crs or manifest_crs or DEFAULT_TARGET_CRS
    annotations = annotations.to_crs(target_crs)
    annotations = annotations[
        annotations.geometry.notna() & ~annotations.geometry.is_empty
    ].copy()
    annotations[args.class_field] = (
        annotations[args.class_field].astype(str).str.strip().str.lower()
    )

    model = YOLO(str(args.weights))
    predictions: list[dict] = []
    results = model.predict(
        source=str(args.images),
        conf=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
    )

    for result in results:
        filename = Path(result.path).name
        if filename not in manifest:
            raise KeyError(f"Image is missing from tile_manifest.csv: {filename}")
        tile = manifest[filename]
        minx = float(tile["minx"])
        miny = float(tile["miny"])
        maxx = float(tile["maxx"])
        maxy = float(tile["maxy"])
        image_height, image_width = result.orig_shape
        if result.boxes is None:
            continue

        xyxy = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        for pixel_box, confidence, raw_class_id in zip(xyxy, confidences, classes):
            x1, y1, x2, y2 = pixel_box
            geo_minx = minx + (x1 / image_width) * (maxx - minx)
            geo_maxx = minx + (x2 / image_width) * (maxx - minx)
            geo_maxy = maxy - (y1 / image_height) * (maxy - miny)
            geo_miny = maxy - (y2 / image_height) * (maxy - miny)
            class_id = int(raw_class_id)
            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": class_name(model, class_id).strip().lower(),
                    "confidence": float(confidence),
                    "tile_name": filename,
                    "split": tile.get("split", ""),
                    "geometry": make_box(geo_minx, geo_miny, geo_maxx, geo_maxy),
                }
            )

    deduplicated = remove_spatial_duplicates(predictions, args.dedupe_iou)
    if not deduplicated:
        raise RuntimeError(
            "The model produced no predictions above --confidence; no GeoPackage was written."
        )
    reviewed: list[dict] = []
    candidates: list[dict] = []

    for number, prediction in enumerate(deduplicated, start=1):
        geometry = prediction["geometry"]
        overlapping = annotations[annotations.geometry.intersects(geometry)]
        best_iou = 0.0
        matched_class = ""
        matched_id = ""
        if not overlapping.empty:
            overlaps = []
            for annotation_id, annotation in overlapping.iterrows():
                overlap = intersection_over_union(geometry, annotation.geometry)
                overlaps.append((overlap, annotation_id, annotation[args.class_field]))
            best_iou, annotation_id, matched_class = max(overlaps, key=lambda item: item[0])
            matched_id = str(annotation_id)

        distances = annotations.geometry.distance(geometry)
        nearest_distance = float(distances.min()) if not distances.empty else float("nan")
        is_candidate = best_iou < args.match_iou
        record = {
            "pred_id": number,
            "class_id": prediction["class_id"],
            "class_name": prediction["class_name"],
            "confidence": round(prediction["confidence"], 6),
            "tile_name": prediction["tile_name"],
            "split": prediction["split"],
            "max_iou": round(best_iou, 6),
            "nearest_m": round(nearest_distance, 3),
            "match_class": str(matched_class),
            "match_id": matched_id,
            "status": "kandidat" if is_candidate else "matcher_annotation",
            "review": "ikke_gennemgaaet" if is_candidate else "",
            "decision": "",
            "comment": "",
            "geometry": geometry,
        }
        reviewed.append(record)
        if is_candidate:
            candidates.append(record.copy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    all_gdf = gpd.GeoDataFrame(reviewed, geometry="geometry", crs=target_crs)
    all_gdf.to_file(args.output, layer="all_predictions", driver="GPKG")
    if candidates:
        candidate_gdf = gpd.GeoDataFrame(candidates, geometry="geometry", crs=target_crs)
        candidate_gdf.to_file(
            args.output, layer="candidates", driver="GPKG", mode="a"
        )

    print("\nPrediction review summary")
    print(f"  Raw predictions: {len(predictions)}")
    print(f"  After removal of tile duplicates: {len(deduplicated)}")
    print(f"  Matching existing annotations: {len(reviewed) - len(candidates)}")
    print(f"  Unmatched review candidates: {len(candidates)}")
    print(f"  Output: {args.output.resolve()}")
    if not candidates:
        print("  No candidates layer was written because every prediction matched an annotation.")


if __name__ == "__main__":
    main()
