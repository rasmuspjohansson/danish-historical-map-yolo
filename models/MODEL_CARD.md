# Model card: `hoje_maalebordsblade_v3.pt`

## Purpose

YOLOv8n object detector trained to locate three cartographic symbols in Danish
historical high-table maps (`Høje målebordsblade`):

1. `engtotter`
2. `mosepolygoner`
3. `vandlinjer`

The additional classes `siv` and `lyng` are distractor classes. They were added
to teach the model not to classify visually similar symbols as `engtotter`.

## Training setup

- Starting checkpoint: the preceding v2 experiment
- Architecture: YOLOv8n detection
- Training software: Ultralytics 8.4.157
- Image size: 640 px
- Epochs: 50
- Device used for this run: CPU
- Training split: 144 tiles
- Validation split: 30 tiles

Tiles overlap, so the number of YOLO instances is larger than the number of
unique source annotations.

## Validation metrics

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 609 | 0.766 | 0.572 | 0.729 | 0.405 |
| engtotter | 104 | 0.718 | 0.906 | 0.893 | 0.488 |
| mosepolygoner | 154 | 0.878 | 0.936 | 0.957 | 0.617 |
| vandlinjer | 291 | 0.688 | 0.893 | 0.886 | 0.415 |
| siv | 2 | 1.000 | 0.000 | 0.695 | 0.426 |
| lyng | 58 | 0.546 | 0.125 | 0.215 | 0.079 |

The `siv` result is not statistically meaningful because the validation split
contains only two instances. `siv` and `lyng` should not be treated as reliable
output classes; their intended role is reducing false positives for the three
target classes.

## Limitations

- The validation areas come from the same annotation project as the training
  areas; results are not an independent national benchmark.
- A `vandlinje` is visually close to other horizontal lines in the map, so
  false positives are expected where context is ambiguous.
- Performance outside the map style, scale, resolution and geographic samples
  represented in the training set is unknown.
- Predictions are review candidates and should not be treated as authoritative
  geographic data without human quality control.

See `docs/assets/` for the training curves, confusion matrix and a validation
prediction mosaic.

SHA-256 of the included weights:

```text
5310c9eb12627aee8f6119fbd45caf0fb14d07936087b7182732af00f91f84d0
```
