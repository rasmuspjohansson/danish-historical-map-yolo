"""Run sliced YOLO inference and write LabelMe-compatible JSON files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tifffile
import torch
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def image_size(path: Path) -> tuple[int, int]:
    """Return image width and height without loading a large TIFF into memory."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            height, width = tif.pages[0].shape[:2]
        return width, height
    with Image.open(path) as image:
        return image.size


def labelme_shape(box, label: str, confidence: float) -> dict:
    x1, y1, x2, y2 = map(float, box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return {
        "label": label,
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "description": f"confidence={confidence:.6f}",
        "shape_type": "rectangle",
        "flags": {},
    }


def labelme_document(image_path: Path, shapes: list[dict]) -> dict:
    width, height = image_size(image_path)
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAHI sliced inference and export one LabelMe JSON per image."
    )
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--folder-with-images", "--folder_with_images", required=True, type=Path)
    parser.add_argument("--result-folder", "--result_folder", required=True, type=Path)
    parser.add_argument("--slice-size", "--slice_width", type=int, default=640)
    parser.add_argument("--overlap-ratio", "--overlap_ratio", type=float, default=0.0625)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument(
        "--device",
        help='Inference device, for example "cpu" or "cuda:0". Auto-selected by default.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.folder_with_images.is_dir():
        raise FileNotFoundError(f"Image folder not found: {args.folder_with_images}")
    if args.slice_size <= 0:
        raise ValueError("--slice-size must be greater than zero")
    if not 0 <= args.overlap_ratio < 1:
        raise ValueError("--overlap-ratio must be at least 0 and smaller than 1")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    args.result_folder.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path
        for path in args.folder_with_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {args.folder_with_images}")

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(args.weights),
        confidence_threshold=args.confidence,
        device=device,
    )

    started = time.time()
    for image_path in tqdm(images, desc="Processing images"):
        result = get_sliced_prediction(
            str(image_path),
            model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap_ratio,
            overlap_width_ratio=args.overlap_ratio,
            verbose=0,
        )
        shapes = [
            labelme_shape(
                prediction.bbox.to_xyxy(),
                prediction.category.name,
                float(prediction.score.value),
            )
            for prediction in result.object_prediction_list
        ]
        destination = args.result_folder / f"{image_path.stem}.json"
        destination.write_text(
            json.dumps(labelme_document(image_path, shapes), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    elapsed = time.time() - started
    print(f"Processed {len(images)} images in {elapsed:.1f} seconds")
    print(f"Results written to: {args.result_folder.resolve()}")


if __name__ == "__main__":
    main()
