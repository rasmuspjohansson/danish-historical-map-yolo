# Data

`annotations/hoje_maalebordsblade_annotations.gpkg` contains the reviewed areas
and bounding-box annotations used for the example model. The relevant layers are:

- `hoje_maalebordsblad_annoteringsomraade`: areas that were reviewed completely.
- `bbox_alle_objekter_v3`: bounding boxes with class names in the `layer` field.

The GeoPackage contains 18 reviewed areas and 2,416 source bounding boxes:

| Class | Source boxes | Role |
| --- | ---: | --- |
| `engtotter` | 841 | target |
| `mosepolygoner` | 474 | target |
| `vandlinjer` | 947 | target |
| `siv` | 60 | helper/distractor |
| `lyng` | 94 | helper/distractor |

Generated TIFF tiles, YOLO text labels and training runs are intentionally not
versioned. Create them with `prepare_wms_yolo_dataset.py`; its default output
can be placed below `data/generated/`, which is ignored by Git.

The map imagery is requested from Klimadatastyrelsen's Datafordeler service at
generation time. No API key is stored in this repository.
