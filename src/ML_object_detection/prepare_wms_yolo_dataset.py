"""Build a YOLO dataset directly from WMS tiles and GeoPackage annotations.

The script reads reviewed annotation polygons and bounding boxes from a
GeoPackage. It creates a regular grid inside the reviewed polygons, downloads
each tile from a WMS as a three-channel RGB PNG, converts intersecting boxes to
YOLO labels, splits by annotation area, and writes dataset.yaml plus a manifest.

The API key is read from an environment variable and is never written to disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import fiona
import geopandas as gpd
from PIL import Image, ImageDraw
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union


DEFAULT_WMS_URL = (
    "https://wms.datafordeler.dk/HoejeMaalebordsblade/"
    "topo20_hoeje_maalebordsblade/1.0.0/wms"
)
DEFAULT_WMS_LAYER = "dtk_hoeje_maalebordsblade"
DEFAULT_CLASSES = ("engtotter", "mosepolygoner", "vandlinjer")
DEFAULT_TARGET_CRS = "EPSG:25832"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download WMS tiles inside reviewed polygons and build a YOLO dataset."
    )
    parser.add_argument("--gpkg", required=True, type=Path, help="Input GeoPackage")
    parser.add_argument("--list-layers", action="store_true", help="List GeoPackage layers and exit")
    parser.add_argument("--areas-layer", help="Polygon layer containing reviewed annotation areas")
    parser.add_argument(
        "--boxes-layer", default="bbox_alle_objekter", help="Bounding-box polygon layer"
    )
    parser.add_argument("--class-field", default="layer", help="Field containing class names")
    parser.add_argument(
        "--area-id-field",
        help="Optional stable ID field in the annotation-area layer",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Class names in numeric YOLO order",
    )
    parser.add_argument("--output", type=Path, help="Output YOLO dataset directory")
    parser.add_argument("--wms-url", default=DEFAULT_WMS_URL, help="WMS endpoint without API key")
    parser.add_argument("--wms-layer", default=DEFAULT_WMS_LAYER, help="WMS layer name")
    parser.add_argument(
        "--api-key-env",
        default="DATAFORDELER_APIKEY",
        help="Environment variable containing the API key",
    )
    parser.add_argument("--wms-version", choices=("1.1.1", "1.3.0"), default="1.1.1")
    parser.add_argument(
        "--target-crs",
        default=DEFAULT_TARGET_CRS,
        help="Projected CRS used for WMS requests and generated tile geometry",
    )
    parser.add_argument("--pixel-size", type=float, default=0.5, help="Ground pixel size in metres")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile width and height in pixels")
    parser.add_argument("--overlap", type=int, default=80, help="Overlap between tiles in pixels")
    parser.add_argument(
        "--min-area-coverage",
        type=float,
        default=1.0,
        help="Required fraction of tile covered by a reviewed area (default: 1.0)",
    )
    parser.add_argument(
        "--min-visible-box",
        type=float,
        default=0.5,
        help="Minimum visible fraction of a box after clipping",
    )
    parser.add_argument(
        "--mask-outside-areas",
        action="store_true",
        help="Replace pixels outside the reviewed annotation polygon with white",
    )
    parser.add_argument(
        "--edge-box-padding-pixels",
        type=int,
        default=0,
        help=(
            "When masking, reveal this many pixels around an annotated box only if "
            "the box crosses the annotation boundary (default: 0; no exception)"
        ),
    )
    parser.add_argument(
        "--include-edge-object-tiles",
        action="store_true",
        help=(
            "Keep a boundary tile below --min-area-coverage when it contains the "
            "centroid of an annotated box inside the reviewed area"
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between WMS requests")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Stop after this many tiles; 0 means no limit",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count eligible tiles without downloading")
    parser.add_argument(
        "--overwrite", action="store_true", help="Redownload existing image tiles"
    )
    return parser.parse_args()


def stable_short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def make_request_url(base_url: str, api_key: str, params: dict[str, str]) -> str:
    parsed = urlparse(base_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    query["apikey"] = api_key
    return urlunparse(parsed._replace(query=urlencode(query)))


def fetch_rgb_tile(
    base_url: str,
    api_key: str,
    wms_layer: str,
    wms_version: str,
    target_crs: str,
    bounds: tuple[float, float, float, float],
    size: int,
    timeout: float,
    retries: int,
) -> Image.Image:
    bbox = ",".join(f"{value:.3f}" for value in bounds)
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": wms_version,
        "LAYERS": wms_layer,
        "STYLES": "",
        "FORMAT": "image/png",
        "TRANSPARENT": "FALSE",
        "BGCOLOR": "0xFFFFFF",
        "BBOX": bbox,
        "WIDTH": str(size),
        "HEIGHT": str(size),
    }
    params["SRS" if wms_version == "1.1.1" else "CRS"] = target_crs
    url = make_request_url(base_url, api_key, params)
    request = Request(url, headers={"User-Agent": "KDS-YOLO-dataset-preparer/1.0"})

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                payload = response.read()
            if not content_type.lower().startswith("image/"):
                message = payload[:500].decode("utf-8", errors="replace")
                raise RuntimeError(f"WMS returned {content_type or 'unknown content'}: {message}")
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            if image.size != (size, size):
                raise RuntimeError(f"WMS returned image size {image.size}, expected {(size, size)}")
            return image
        except (HTTPError, URLError, OSError, RuntimeError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"WMS download failed after {retries} attempts: {last_error}")


def choose_validation_areas(area_keys: list[str], fraction: float, seed: int) -> set[str]:
    unique = sorted(set(area_keys))
    if len(unique) < 2:
        return set()
    shuffled = unique.copy()
    random.Random(seed).shuffle(shuffled)
    count = max(1, min(len(unique) - 1, round(len(unique) * fraction)))
    return set(shuffled[:count])


def iter_grid(area_geometry, tile_ground_size: float, step_ground_size: float):
    minx, miny, maxx, maxy = area_geometry.bounds
    start_x = math.floor(minx / step_ground_size) * step_ground_size
    start_y = math.floor(miny / step_ground_size) * step_ground_size
    x = start_x
    while x < maxx:
        y = start_y
        while y < maxy:
            yield (x, y, x + tile_ground_size, y + tile_ground_size)
            y += step_ground_size
        x += step_ground_size


def yolo_lines_for_tile(
    boxes: gpd.GeoDataFrame,
    tile_geometry,
    class_to_id: dict[str, int],
    class_field: str,
    min_visible: float,
    reviewed_geometry,
) -> tuple[list[str], dict[str, int], list[object], list[str]]:
    minx, miny, maxx, maxy = tile_geometry.bounds
    tile_width = maxx - minx
    tile_height = maxy - miny
    candidates = boxes[boxes.geometry.intersects(tile_geometry)]
    lines: list[str] = []
    counts = {name: 0 for name in class_to_id}
    included_geometries: list[object] = []
    included_ids: list[str] = []

    for feature_id, feature in candidates.iterrows():
        geometry = feature.geometry
        if geometry is None or geometry.is_empty or geometry.area <= 0:
            continue
        if not tile_geometry.covers(geometry.centroid):
            continue
        if not reviewed_geometry.covers(geometry.centroid):
            continue
        clipped = geometry.intersection(tile_geometry)
        if clipped.is_empty or clipped.area / geometry.area < min_visible:
            continue

        bx_min, by_min, bx_max, by_max = clipped.bounds
        x_center = (((bx_min + bx_max) / 2) - minx) / tile_width
        y_center = (maxy - ((by_min + by_max) / 2)) / tile_height
        width = (bx_max - bx_min) / tile_width
        height = (by_max - by_min) / tile_height

        class_name = str(feature[class_field]).strip().lower()
        class_id = class_to_id[class_name]
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
        counts[class_name] += 1
        included_geometries.append(geometry)
        included_ids.append(str(feature_id))
    return lines, counts, included_geometries, included_ids


def _draw_geometry(draw: ImageDraw.ImageDraw, geometry, bounds, size: int) -> None:
    """Rasterize Polygon/MultiPolygon geometry into a PIL mask."""
    minx, miny, maxx, maxy = bounds

    def pixel_ring(coords):
        return [
            (
                (x - minx) / (maxx - minx) * size,
                (maxy - y) / (maxy - miny) * size,
            )
            for x, y in coords
        ]

    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        draw.polygon(pixel_ring(geometry.exterior.coords), fill=255)
        for interior in geometry.interiors:
            draw.polygon(pixel_ring(interior.coords), fill=0)
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geometry.geoms:
            _draw_geometry(draw, part, bounds, size)


def mask_image_outside_geometry(
    image: Image.Image,
    bounds: tuple[float, float, float, float],
    reviewed_geometry,
    included_boxes: list[object],
    edge_padding_ground: float,
) -> tuple[Image.Image, float, int]:
    """Mask unknown pixels and optionally preserve local context at cut boxes."""
    tile_geometry = shapely_box(*bounds)
    mask_geometry = reviewed_geometry.intersection(tile_geometry)
    rescued = 0
    if edge_padding_ground > 0:
        edge_boxes = [g for g in included_boxes if not reviewed_geometry.covers(g)]
        if edge_boxes:
            local_context = unary_union(
                [g.buffer(edge_padding_ground, cap_style=3, join_style=2) for g in edge_boxes]
            )
            mask_geometry = mask_geometry.union(local_context.intersection(tile_geometry))
            rescued = len(edge_boxes)

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    _draw_geometry(draw, mask_geometry, bounds, image.size[0])
    white = Image.new("RGB", image.size, (255, 255, 255))
    masked = Image.composite(image, white, mask)
    histogram = mask.histogram()
    total_pixels = image.size[0] * image.size[1]
    visible_fraction = sum(i * count for i, count in enumerate(histogram)) / (255 * total_pixels)
    return masked, 1.0 - visible_fraction, rescued


def write_dataset_yaml(output: Path, classes: list[str]) -> None:
    lines = [
        f'path: "{output.resolve().as_posix()}"',
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(classes))
    (output / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.gpkg.is_file():
        raise FileNotFoundError(f"GeoPackage not found: {args.gpkg}")

    if args.list_layers:
        print("Layers in GeoPackage:")
        for layer_name in fiona.listlayers(str(args.gpkg)):
            print(f"  {layer_name}")
        return

    if not args.areas_layer:
        raise ValueError("--areas-layer is required (use --list-layers to see layer names)")
    if not args.output:
        raise ValueError("--output is required")
    if args.tile_size <= 0 or args.pixel_size <= 0:
        raise ValueError("--tile-size and --pixel-size must be greater than zero")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("--overlap must be at least 0 and smaller than --tile-size")
    if args.edge_box_padding_pixels < 0:
        raise ValueError("--edge-box-padding-pixels must be 0 or greater")
    if args.edge_box_padding_pixels and not args.mask_outside_areas:
        raise ValueError(
            "--edge-box-padding-pixels requires --mask-outside-areas"
        )
    for name, value in (
        ("--min-area-coverage", args.min_area_coverage),
        ("--min-visible-box", args.min_visible_box),
        ("--val-fraction", args.val_fraction),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and not args.dry_run:
        raise RuntimeError(
            f"API key not found. Set environment variable {args.api_key_env} first."
        )

    areas = gpd.read_file(args.gpkg, layer=args.areas_layer)
    boxes = gpd.read_file(args.gpkg, layer=args.boxes_layer)
    if areas.empty:
        raise ValueError("The annotation-area layer is empty")
    if areas.crs is None or boxes.crs is None:
        raise ValueError("Both layers must have a CRS")
    if args.class_field not in boxes.columns:
        raise KeyError(f"Class field '{args.class_field}' not found in {args.boxes_layer}")
    if args.area_id_field and args.area_id_field not in areas.columns:
        raise KeyError(f"Area ID field '{args.area_id_field}' not found in {args.areas_layer}")

    areas = areas.to_crs(args.target_crs)
    boxes = boxes.to_crs(args.target_crs)
    areas = areas[areas.geometry.notna() & ~areas.geometry.is_empty].copy()
    boxes = boxes[boxes.geometry.notna() & ~boxes.geometry.is_empty].copy()
    boxes[args.class_field] = boxes[args.class_field].astype(str).str.strip().str.lower()

    classes = [name.strip().lower() for name in args.classes]
    if len(classes) != len(set(classes)):
        raise ValueError("Class names must be unique")
    class_to_id = {name: index for index, name in enumerate(classes)}
    unknown = sorted(set(boxes[args.class_field]) - set(class_to_id))
    if unknown:
        raise ValueError("Unknown class name(s) in boxes layer: " + ", ".join(unknown))

    if args.area_id_field:
        areas["_area_key"] = areas[args.area_id_field].astype(str)
    else:
        areas["_area_key"] = [f"area_{index:04d}" for index in range(len(areas))]
    validation_areas = choose_validation_areas(
        areas["_area_key"].tolist(), args.val_fraction, args.seed
    )
    single_area_mode = len(set(areas["_area_key"])) < 2

    tile_ground_size = args.tile_size * args.pixel_size
    step_ground_size = (args.tile_size - args.overlap) * args.pixel_size
    output = args.output
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_path = output / "tile_manifest.csv"
    manifest_rows: list[dict[str, str]] = []
    tile_records: list[dict[str, object]] = []
    seen_tiles: set[tuple[int, int]] = set()
    image_counts = {"train": 0, "val": 0}
    object_counts = {
        "train": {name: 0 for name in classes},
        "val": {name: 0 for name in classes},
    }

    stop = False
    for _, area_feature in areas.iterrows():
        area_geometry = area_feature.geometry
        area_key = str(area_feature["_area_key"])
        area_hash = stable_short_hash(area_key)
        area_minx, _, area_maxx, _ = area_geometry.bounds
        single_area_val_start = area_minx + (area_maxx - area_minx) * (1 - args.val_fraction)

        for bounds in iter_grid(area_geometry, tile_ground_size, step_ground_size):
            tile_geometry = shapely_box(*bounds)
            coverage = tile_geometry.intersection(area_geometry).area / tile_geometry.area
            has_reviewed_object = False
            if args.include_edge_object_tiles:
                nearby_boxes = boxes[boxes.geometry.intersects(tile_geometry)]
                has_reviewed_object = any(
                    area_geometry.covers(geometry.centroid)
                    and tile_geometry.covers(geometry.centroid)
                    for geometry in nearby_boxes.geometry
                    if geometry is not None and not geometry.is_empty
                )
            if coverage + 1e-9 < args.min_area_coverage and not has_reviewed_object:
                continue

            grid_key = (
                round(bounds[0] / args.pixel_size),
                round(bounds[1] / args.pixel_size),
            )
            if grid_key in seen_tiles:
                continue
            seen_tiles.add(grid_key)

            if single_area_mode:
                split = "val" if tile_geometry.centroid.x >= single_area_val_start else "train"
            else:
                split = "val" if area_key in validation_areas else "train"

            east_code = int(round(bounds[0] / args.pixel_size))
            north_code = int(round(bounds[1] / args.pixel_size))
            stem = f"area_{area_hash}_e{east_code}_n{north_code}"
            image_path = output / "images" / split / f"{stem}.png"
            label_path = output / "labels" / split / f"{stem}.txt"

            lines, tile_class_counts, included_boxes, included_ids = yolo_lines_for_tile(
                boxes,
                tile_geometry,
                class_to_id,
                args.class_field,
                args.min_visible_box,
                area_geometry,
            )

            masked_fraction = 0.0
            rescued_edge_boxes = 0
            if not args.dry_run and (args.overwrite or not image_path.exists()):
                image = fetch_rgb_tile(
                    args.wms_url,
                    api_key,
                    args.wms_layer,
                    args.wms_version,
                    args.target_crs,
                    bounds,
                    args.tile_size,
                    args.timeout,
                    args.retries,
                )
                if args.mask_outside_areas:
                    image, masked_fraction, rescued_edge_boxes = mask_image_outside_geometry(
                        image,
                        bounds,
                        area_geometry,
                        included_boxes,
                        args.edge_box_padding_pixels * args.pixel_size,
                    )
                image.save(image_path, format="PNG", optimize=True)
                if args.delay > 0:
                    time.sleep(args.delay)
            if not args.dry_run:
                label_path.write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )

            image_counts[split] += 1
            for class_name, count in tile_class_counts.items():
                object_counts[split][class_name] += count
            class_count_fields = {
                f"class_{class_id}_count": str(tile_class_counts.get(class_name, 0))
                for class_name, class_id in class_to_id.items()
            }
            manifest_row = {
                    "filename": f"{stem}.png",
                    "split": split,
                    "area_id": area_key,
                    "crs": args.target_crs,
                    "minx": f"{bounds[0]:.3f}",
                    "miny": f"{bounds[1]:.3f}",
                    "maxx": f"{bounds[2]:.3f}",
                    "maxy": f"{bounds[3]:.3f}",
                    "coverage": f"{coverage:.6f}",
                    "masked_fraction": f"{masked_fraction:.6f}",
                    "objects": str(len(lines)),
                    "edge_boxes_rescued": str(rescued_edge_boxes),
                    "edge_tile_kept": str(
                        coverage + 1e-9 < args.min_area_coverage and has_reviewed_object
                    ),
                    "source_box_ids": ";".join(included_ids),
                }
            manifest_row.update(class_count_fields)
            manifest_rows.append(manifest_row)

            tile_record = {
                    "tile_name": f"{stem}.png",
                    "split": split,
                    "area_id": area_key,
                    "coverage": coverage,
                    "masked_pct": masked_fraction * 100,
                    "objects": len(lines),
                    "edge_saved": rescued_edge_boxes,
                    "edge_tile": coverage + 1e-9 < args.min_area_coverage
                    and has_reviewed_object,
                    "geometry": tile_geometry,
                }
            tile_record.update(
                {
                    f"class_{class_id}": tile_class_counts.get(class_name, 0)
                    for class_name, class_id in class_to_id.items()
                }
            )
            tile_records.append(tile_record)

            total = image_counts["train"] + image_counts["val"]
            if total % 25 == 0:
                print(f"Prepared {total} tiles...")
            if args.max_tiles and total >= args.max_tiles:
                stop = True
                break
        if stop:
            break

    if not manifest_rows:
        raise RuntimeError(
            "No eligible tiles were found. The annotation polygons may be smaller than "
            "one tile; try a smaller --tile-size or lower --min-area-coverage."
        )

    if not args.dry_run:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
        tiles_gdf = gpd.GeoDataFrame(tile_records, geometry="geometry", crs=args.target_crs)
        tiles_gdf.to_file(
            output / "generated_tiles.gpkg",
            layer="generated_tiles",
            driver="GPKG",
        )
        write_dataset_yaml(output, classes)
        generation_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        (output / "generation_config.json").write_text(
            json.dumps(generation_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("\nDataset summary" if not args.dry_run else "\nDry-run summary")
    for split in ("train", "val"):
        print(f"  {split}: {image_counts[split]} images")
        for class_name in classes:
            print(f"    {class_name}: {object_counts[split][class_name]} boxes")
    if image_counts["train"] == 0 or image_counts["val"] == 0:
        print("WARNING: one split has no images; add more areas or adjust --val-fraction.")
    for split in ("train", "val"):
        if sum(object_counts[split].values()) == 0:
            print(f"WARNING: {split} has no positive object labels.")
    if args.dry_run:
        print("No images or labels were written because --dry-run was used.")
    else:
        print(f"Dataset written to: {output.resolve()}")
        print(f"Training config: {output.resolve() / 'dataset.yaml'}")
        print(f"QGIS tile overview: {output.resolve() / 'generated_tiles.gpkg'}")


if __name__ == "__main__":
    main()
