"""Create YOLO label files for georeferenced image tiles from GeoPackage boxes."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box


DEFAULT_CLASSES = ("engtotter", "mosepolygoner", "vandlinjer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert bounding boxes in a GeoPackage to one YOLO .txt label "
            "file per georeferenced raster tile."
        )
    )
    parser.add_argument("--tiles", required=True, type=Path, help="Folder with GeoTIFF tiles")
    parser.add_argument("--gpkg", required=True, type=Path, help="GeoPackage containing bounding boxes")
    parser.add_argument(
        "--layer",
        default="bbox_alle_objekter",
        help="GeoPackage layer name (default: bbox_alle_objekter)",
    )
    parser.add_argument(
        "--class-field",
        default="layer",
        help="Attribute containing class names (default: layer)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Folder for YOLO .txt files")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Class names in numeric YOLO order",
    )
    parser.add_argument(
        "--min-visible",
        type=float,
        default=0.5,
        help="Minimum visible fraction of a box after clipping (default: 0.5)",
    )
    return parser.parse_args()


def geometry_to_yolo(geometry, transform, width: int, height: int):
    """Return normalized x-center, y-center, width and height for a geometry."""
    xmin, ymin, xmax, ymax = geometry.bounds
    inverse = ~transform
    pixel_corners = [
        inverse * (xmin, ymin),
        inverse * (xmin, ymax),
        inverse * (xmax, ymin),
        inverse * (xmax, ymax),
    ]
    columns = [point[0] for point in pixel_corners]
    rows = [point[1] for point in pixel_corners]

    left = max(0.0, min(columns))
    right = min(float(width), max(columns))
    top = max(0.0, min(rows))
    bottom = min(float(height), max(rows))

    box_width = right - left
    box_height = bottom - top
    if box_width <= 0 or box_height <= 0:
        return None

    return (
        ((left + right) / 2.0) / width,
        ((top + bottom) / 2.0) / height,
        box_width / width,
        box_height / height,
    )


def main() -> None:
    args = parse_args()

    if not args.tiles.is_dir():
        raise FileNotFoundError(f"Tile folder not found: {args.tiles}")
    if not args.gpkg.is_file():
        raise FileNotFoundError(f"GeoPackage not found: {args.gpkg}")
    if not 0 <= args.min_visible <= 1:
        raise ValueError("--min-visible must be between 0 and 1")

    annotations = gpd.read_file(args.gpkg, layer=args.layer)
    if args.class_field not in annotations.columns:
        raise KeyError(
            f"Field '{args.class_field}' was not found. Available fields: "
            f"{', '.join(map(str, annotations.columns))}"
        )
    if annotations.crs is None:
        raise ValueError("The annotation layer has no coordinate reference system (CRS)")

    annotations = annotations[annotations.geometry.notna() & ~annotations.geometry.is_empty].copy()
    annotations["_class_name"] = (
        annotations[args.class_field].astype(str).str.strip().str.lower()
    )

    classes = [name.strip().lower() for name in args.classes]
    if len(classes) != len(set(classes)):
        raise ValueError("Class names must be unique")
    class_to_id = {name: index for index, name in enumerate(classes)}
    unknown = sorted(set(annotations["_class_name"]) - set(class_to_id))
    if unknown:
        raise ValueError(
            "Unknown class name(s) in the annotation layer: " + ", ".join(unknown)
        )

    tiles = sorted(
        path for path in args.tiles.iterdir() if path.suffix.lower() in {".tif", ".tiff"}
    )
    if not tiles:
        raise FileNotFoundError(f"No .tif or .tiff tiles found in: {args.tiles}")

    args.output.mkdir(parents=True, exist_ok=True)
    annotation_crs = annotations.crs
    labels_written = 0
    objects_written = 0

    for tile_path in tiles:
        with rasterio.open(tile_path) as raster:
            if raster.crs is None:
                raise ValueError(f"Tile has no CRS: {tile_path}")

            tile_annotations = (
                annotations
                if annotation_crs == raster.crs
                else annotations.to_crs(raster.crs)
            )
            tile_extent = box(*raster.bounds)
            candidates = tile_annotations[
                tile_annotations.geometry.intersects(tile_extent)
            ]

            lines = []
            for _, feature in candidates.iterrows():
                geometry = feature.geometry

                # With overlapping tiles an object may legitimately occur in more
                # than one tile. Requiring its center to be inside the tile avoids
                # labels made only from a tiny sliver at a tile edge.
                if not tile_extent.covers(geometry.centroid):
                    continue

                clipped = geometry.intersection(tile_extent)
                if clipped.is_empty or geometry.area <= 0:
                    continue
                if clipped.area / geometry.area < args.min_visible:
                    continue

                yolo_box = geometry_to_yolo(
                    clipped, raster.transform, raster.width, raster.height
                )
                if yolo_box is None:
                    continue

                class_id = class_to_id[feature["_class_name"]]
                x_center, y_center, box_width, box_height = yolo_box
                lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{box_width:.6f} {box_height:.6f}"
                )

        label_path = args.output / f"{tile_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        labels_written += 1
        objects_written += len(lines)

    print(f"Created {labels_written} label files in '{args.output}'.")
    print(f"Wrote {objects_written} object annotations.")
    print(
        "Class mapping: "
        + ", ".join(f"{class_id}={name}" for name, class_id in class_to_id.items())
    )


if __name__ == "__main__":
    main()
