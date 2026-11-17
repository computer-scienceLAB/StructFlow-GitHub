import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

from evaluation import evaluate
from inference.core import sha256
from .checkpoints import ROOT, validate

DEPTH_KEYS = {'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'delta1', 'delta2', 'delta3', 'valid_depth_pixels'}
DISPARITY_KEYS = {'disp_mae', 'disp_d1_bad3', 'valid_disparity_pixels'}
FLOW_KEYS = {'epe_all', 'epe_match', 'epe_unmatch', 'valid_flow_pixels', 'unmatched_valid_percent'}


def select_keys(task, dataset, protocol):
    if task == 'depth':
        return DEPTH_KEYS if dataset in ('kitti2015', 'eigen') else DISPARITY_KEYS
    return FLOW_KEYS | ({'f1_9pct'} if protocol == 'source-9pct' else {'f1_5pct'})


def main(task):
    p = argparse.ArgumentParser(description=f'StructFlow {task}-only evaluation; uses the unchanged release evaluator.')
    p.add_argument('--dataset', required=True, choices=(['kitti2015', 'kitti2012', 'sintel', 'flyingthings3d', 'spring']
                                                     if task == 'flow' else ['kitti2015', 'eigen', 'flyingthings3d', 'spring']))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--split', type=Path)
    p.add_argument('--flow-root', type=Path)
    p.add_argument('--allow-candidate', action='store_true')
    p.add_argument('--limit', type=int, help='Explicit smoke-test prefix, never a full benchmark result.')
    p.add_argument('--full', action='store_true')
    p.add_argument('--model-implementation', choices=['release', 'historical-final'], default='release')
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--visualize', action='store_true', help='Render task-only prediction maps; implies --save-predictions.')
    if task == 'flow':
        p.add_argument('--metric-protocol', choices=['kitti-standard', 'source-9pct'], default='kitti-standard')
    a = p.parse_args()
    protocol = getattr(a, 'metric_protocol', None)
    try:
        name, checksum = validate(a.checkpoint, task, a.dataset, a.allow_candidate)
    except ValueError as exc:
        p.error(str(exc))
    split_name = {'eigen': 'kitti_eigen_697.txt', 'flyingthings3d': 'flyingthings3d_val_500.txt'}.get(a.dataset)
    if split_name:
        expected = ROOT / 'core' / split_name
        a.split = a.split or expected
        if not a.split.is_file() or sha256(a.split) != sha256(expected):
            p.error('Use the unchanged release split; a different subset is not this reproduction protocol.')
    elif a.split:
        p.error('--split is only supported for Eigen and FlyingThings3D.')
    if a.dataset == 'sintel' and a.flow_root is None:
        p.error('Sintel requires --flow-root for the official forward-flow training annotations.')
    if a.full and a.limit is not None:
        p.error('--full and --limit cannot be used together.')
    if a.dataset == 'spring' and a.limit is None and not a.full:
        a.limit = 200
    args = ['--dataset', a.dataset, '--root', str(a.root), '--checkpoint', str(a.checkpoint),
            '--task', task, '--output', str(a.output), '--device', a.device,
            '--model-implementation', a.model_implementation]
    for flag, value in [('--split', a.split), ('--flow-root', a.flow_root), ('--limit', a.limit)]:
        if value is not None:
            args += [flag, str(value)]
    if a.save_predictions or a.visualize:
        args += ['--save-predictions']
    old_argv = sys.argv
    try:
        sys.argv = ['evaluation.evaluate', *args]
        # Expose only the metrics for the selected task.
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate.main()
    finally:
        sys.argv = old_argv
    summary_path = a.output / 'summary.json'
    summary = json.loads(summary_path.read_text())
    keep = select_keys(task, a.dataset, protocol)
    summary = {k: v for k, v in summary.items() if k in keep}
    if task == 'flow':
        summary['f1'] = summary.pop('f1_9pct' if protocol == 'source-9pct' else 'f1_5pct')
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    sample_path = a.output / 'per_sample.csv'
    if sample_path.is_file():
        with sample_path.open() as f:
            reader = csv.DictReader(f)
            keys = [key for key in reader.fieldnames if key == 'sample_id' or key in keep]
            rows = [{key: row[key] for key in keys} for row in reader]
        if task == 'flow':
            metric_key = 'f1_9pct' if protocol == 'source-9pct' else 'f1_5pct'
            keys[keys.index(metric_key)] = 'f1'
            for row in rows:
                row['f1'] = row.pop(metric_key)
        with sample_path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    if a.visualize:
        render_predictions(a, task)
    print(json.dumps(summary, indent=2, allow_nan=False))


def render_predictions(args, task):
    import numpy as np
    from PIL import Image
    from inference.core import calibration
    from inference.visualization import flow_rgb, scalar_rgb, save_legend
    records = json.loads((args.output / 'sample_manifest.json').read_text())
    quantity = 'flow' if task == 'flow' else 'depth' if args.dataset in ('kitti2015', 'eigen') else 'disparity'
    out = args.output / 'visualizations'; out.mkdir()
    for i, row in enumerate(records):
        directory = args.output / 'predictions' / f'{i:06d}'
        value = np.load(directory / ('flow.npy' if task == 'flow' else 'disparity.npy'))
        if quantity == 'depth':
            root_label = {'kitti2015': 'KITTI2015/training', 'eigen': 'KITTI-Raw'}[args.dataset]
            calibration_path = str(row['calibration'])
            if calibration_path.startswith(root_label + '/'):
                calibration_path = calibration_path[len(root_label) + 1:]
            f, b = calibration(args.root / calibration_path)
            value = (f * b / np.maximum(value, 1e-3)).astype(np.float32)
            np.save(directory / 'depth.npy', value)
        pixels = flow_rgb(value) if task == 'flow' else scalar_rgb(value, 80)
        Image.fromarray(pixels).save(out / f'{i:06d}_{quantity}.png')
    save_legend(out / 'legend.png', 'flow' if task == 'flow' else quantity + (' (m)' if quantity == 'depth' else ' (px)'), 80)
