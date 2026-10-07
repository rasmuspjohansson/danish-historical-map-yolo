"""Export one-to-one detection outcomes to a GeoPackage.

Each annotation inside the evaluated tiles can be claimed by only one
prediction. Predictions are matched in confidence order to the unused
annotation with the greatest overlap. The output layers are true_class,
wrong_class, false_positives, and false_negatives.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box as make_box
from shapely.ops import unary_union
from ultralytics import YOLO


DEFAULT_TARGET_CRS = "EPSG:25832"
LAYER_NAMES = ("true_class", "wrong_class", "false_positives", "false_negatives")
PREDICTION_COLUMNS = (
    "pred_id",
    "class_name",
    "confidence",
    "tile_name",
    "annotation_id",
    "annotation_class",
    "iou",
)
ANNOTATION_COLUMNS = ("annotation_id", "class_name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare detections with annotations and write true class, wrong class, "
            "false positive, and false negative layers."
        )
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
        help="Minimum overlap required to assign a prediction to an annotation",
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
    return str(names[class_id]).strip().lower()


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


def prediction_frame(records: list[dict], crs: str) -> gpd.GeoDataFrame:
    frame = pd.DataFrame(records, columns=list(PREDICTION_COLUMNS))
    geometries = [record["geometry"] for record in records]
    return gpd.GeoDataFrame(frame, geometry=geometries, crs=crs)


def annotation_frame(records: list[dict], crs: str) -> gpd.GeoDataFrame:
    frame = pd.DataFrame(records, columns=list(ANNOTATION_COLUMNS))
    geometries = [record["geometry"] for record in records]
    return gpd.GeoDataFrame(frame, geometry=geometries, crs=crs)


def write_layers(path: Path, layers: dict[str, gpd.GeoDataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    first = True
    for name in LAYER_NAMES:
        layers[name].to_file(
            path,
            layer=name,
            driver="GPKG",
            mode="w" if first else "a",
        )
        first = False


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
    for name, value in (
        ("--confidence", args.confidence),
        ("--match-iou", args.match_iou),
        ("--dedupe-iou", args.dedupe_iou),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --overwrite to replace it."
        )

    manifest = read_manifest(args.manifest)
    image_names = {
        path.name
        for path in args.images.iterdir()
        if path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    }
    if not image_names:
        raise FileNotFoundError(f"No images found in {args.images}")
    missing_manifest = sorted(name for name in image_names if name not in manifest)
    if missing_manifest:
        raise KeyError(
            "Images missing from tile_manifest.csv: " + ", ".join(missing_manifest[:5])
        )

    manifest_crs_values = {
        row.get("crs", "").strip()
        for row in manifest.values()
        if row.get("crs", "").strip()
    }
    if len(manifest_crs_values) > 1:
        raise ValueError("Manifest contains more than one CRS")
    manifest_crs = next(iter(manifest_crs_values), "")
    target_crs = args.target_crs or manifest_crs or DEFAULT_TARGET_CRS

    tile_geometries = []
    for name in image_names:
        tile = manifest[name]
        tile_geometries.append(
            make_box(
                float(tile["minx"]),
                float(tile["miny"]),
                float(tile["maxx"]),
                float(tile["maxy"]),
            )
        )
    evaluated_area = unary_union(tile_geometries)

    annotations = gpd.read_file(args.annotations_gpkg, layer=args.annotations_layer)
    if annotations.crs is None:
        raise ValueError("The annotation layer has no CRS")
    if args.class_field not in annotations.columns:
        raise KeyError(f"Field '{args.class_field}' is missing from the annotation layer")
    annotations = annotations.to_crs(target_crs)
    annotations = annotations[
        annotations.geometry.notna() & ~annotations.geometry.is_empty
    ].copy()
    annotations[args.class_field] = (
        annotations[args.class_field].astype(str).str.strip().str.lower()
    )
    eligible = annotations[annotations.geometry.centroid.within(evaluated_area)]

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
        tile = manifest[filename]
        minx = float(tile["minx"])
        miny = float(tile["miny"])
        maxx = float(tile["maxx"])
        maxy = float(tile["maxy"])
        image_height, image_width = result.orig_shape
        if result.boxes is None:
            continue
        for pixel_box, confidence, raw_class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            x1, y1, x2, y2 = pixel_box
            class_id = int(raw_class_id)
            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": class_name(model, class_id),
                    "confidence": float(confidence),
                    "tile_name": filename,
                    "geometry": make_box(
                        minx + (x1 / image_width) * (maxx - minx),
                        maxy - (y2 / image_height) * (maxy - miny),
                        minx + (x2 / image_width) * (maxx - minx),
                        maxy - (y1 / image_height) * (maxy - miny),
                    ),
                }
            )

    deduplicated = remove_spatial_duplicates(predictions, args.dedupe_iou)
    annotation_rows = [
        (str(index), row.geometry, row[args.class_field])
        for index, row in eligible.iterrows()
    ]
    used_annotations: set[str] = set()
    true_class: list[dict] = []
    wrong_class: list[dict] = []
    false_positives: list[dict] = []

    for number, prediction in enumerate(
        sorted(deduplicated, key=lambda item: item["confidence"], reverse=True),
        start=1,
    ):
        best_id = ""
        best_class = ""
        best_iou = 0.0
        for annotation_id, geometry, annotation_class in annotation_rows:
            if annotation_id in used_annotations:
                continue
            overlap = intersection_over_union(prediction["geometry"], geometry)
            if overlap > best_iou:
                best_iou = overlap
                best_id = annotation_id
                best_class = str(annotation_class)
        record = {
            "pred_id": number,
            "class_name": prediction["class_name"],
            "confidence": round(prediction["confidence"], 6),
            "tile_name": prediction["tile_name"],
            "annotation_id": "",
            "annotation_class": "",
            "iou": 0.0,
            "geometry": prediction["geometry"],
        }
        if best_id and best_iou >= args.match_iou:
            used_annotations.add(best_id)
            record["annotation_id"] = best_id
            record["annotation_class"] = best_class
            record["iou"] = round(best_iou, 6)
            if prediction["class_name"] == best_class:
                true_class.append(record)
            else:
                wrong_class.append(record)
        else:
            false_positives.append(record)

    false_negatives = [
        {
            "annotation_id": annotation_id,
            "class_name": str(annotation_class),
            "geometry": geometry,
        }
        for annotation_id, geometry, annotation_class in annotation_rows
        if annotation_id not in used_annotations
    ]

    layers = {
        "true_class": prediction_frame(true_class, target_crs),
        "wrong_class": prediction_frame(wrong_class, target_crs),
        "false_positives": prediction_frame(false_positives, target_crs),
        "false_negatives": annotation_frame(false_negatives, target_crs),
    }
    write_layers(args.output, layers)

    print("\nDetection comparison")
    print(f"  Raw predictions: {len(predictions)}")
    print(f"  After removal of tile duplicates: {len(deduplicated)}")
    print(f"  Annotations inside evaluated tiles: {len(annotation_rows)}")
    for name in LAYER_NAMES:
        print(f"  {name}: {len(layers[name])}")
    print(f"  Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
