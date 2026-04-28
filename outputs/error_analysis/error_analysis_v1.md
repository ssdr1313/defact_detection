# YOLO Class-Level Error Analysis

## Settings

- images: `/home/kdy/pycharm_prj/defact_detection/data/yolo/images/test`
- ground truth labels: `/home/kdy/pycharm_prj/defact_detection/data/yolo/labels/test`
- prediction labels: `/home/kdy/pycharm_prj/defact_detection/runs/detect/neu_baseline_v1_pred-3/labels`
- IoU threshold for TP: `0.5`
- IoU range for bad localization: `[0.1, 0.5)`
- prediction confidence filter: `0.0`

## Overall Counts

- GT instances: `431`
- TP: `371`
- FP: `435`
- FN: `60`
- Class errors: `2`
- Bad localization cases: `41`

## Error Type Counts

- bad_location: `41`
- class_error: `2`
- false_negative: `17`
- false_positive: `392`

## Class-Level Summary

| class_name | gt_instances | TP | FP | FN | precision_by_matching | recall_by_matching | class_error_as_gt | bad_location |
|---|---|---|---|---|---|---|---|---|
| crazing | 80 | 47 | 127 | 33 | 0.270115 | 0.587500 | 0 | 28 |
| inclusion | 92 | 86 | 84 | 6 | 0.505882 | 0.934783 | 0 | 3 |
| patches | 99 | 96 | 47 | 3 | 0.671329 | 0.969697 | 0 | 0 |
| pitted_surface | 36 | 32 | 36 | 4 | 0.470588 | 0.888889 | 2 | 1 |
| rolled-in_scale | 60 | 50 | 106 | 10 | 0.320513 | 0.833333 | 0 | 7 |
| scratches | 64 | 60 | 35 | 4 | 0.631579 | 0.937500 | 0 | 2 |

## Weak Classes by Recall

- `crazing`: recall=0.587500, FN=33, class_error_as_gt=0, bad_location=28
- `rolled-in_scale`: recall=0.833333, FN=10, class_error_as_gt=0, bad_location=7
- `pitted_surface`: recall=0.888889, FN=4, class_error_as_gt=2, bad_location=1

## Output Files

- `class_error_summary.csv`: class-level TP/FP/FN table.
- `error_cases.csv`: detailed FP/FN/class-error/bad-localization cases.
- `gt_visualization/`: ground-truth-only visualization images.
- `visual_cases/`: side-by-side GT / prediction / overlay panels for error cases.
