# Core

- `evaluate.py` - mapped full-table evaluation, including task-separated Table VI Spring commands; prints and saves computed metrics.
- `evaluate_boundary.py` - reproduces Table XII ground-truth boundary regions and delegates Table XIII's recovered 40-sample GSPN/SFIN inference, saving calculated metrics under `outputs/`.
- `check_spring_registry.py` - verifies that Spring table commands retain their checkpoint, model implementation, and dataset bindings.
- `visualize.py` - user-selected stereo-sequence flow/disparity PNG and MP4 renderer; `--task combined` creates an `Original | Flow | Disparity` video from separate `--flow-checkpoint` and `--disp-checkpoint` arguments. `--overwrite` reruns a visualization in an existing output directory. It accepts strict KITTI benchmark pairs, contiguous KITTI Raw sequences, and matching Spring frame IDs.
- `evaluate_flow.py`, `evaluate_depth.py` - task-specific evaluators.
- `infer_flow.py`, `infer_depth.py` - task-specific inference.
- `models/` - StructFlow model definitions and required operators.
- `inference/`, `evaluation/` - preprocessing, metrics, formats, and visualization.
- `examples/`, `demo/` - qualitative figures and synchronized videos; `examples/kitti2012_sample.json` supplies the KITTI 2012 flow sample manifest.
- `../tables-XII-XIII/` - boundary/GSPN diagnostic for Tables XII and XIII.
