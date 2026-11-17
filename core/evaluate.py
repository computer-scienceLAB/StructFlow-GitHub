"""Evaluate a mapped StructFlow experiment and print/save calculated metrics."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_paths import FLYINGTHINGS3D, KITTI_2012, KITTI_2015, KITTI_RAW, SINTEL, SINTEL_FLOW, SPRING_ADAPTED, SPRING_FULL

EXPERIMENTS = {
    'II': ('depth', 'kitti2015', 'checkpoints/main-results/depth/kitti15-main-depth.pth.tar', KITTI_2015, []),
    'III': ('depth', 'eigen', 'checkpoints/main-results/depth/kitti15-main-depth.pth.tar', KITTI_RAW, ['--split', 'core/kitti_eigen_697.txt']),
    'IV': ('flow', 'kitti2015', 'checkpoints/main-results/flow/kitti15-main-flow.pth.tar', KITTI_2015, ['--metric-protocol', 'source-9pct']),
    'IV-KITTI2012': ('flow', 'kitti2012', 'checkpoints/main-results/flow/kitti12-main-flow.pth.tar', KITTI_2012, ['--metric-protocol', 'source-9pct', '--allow-candidate']),
    'V': ('flow', 'sintel', 'checkpoints/main-results/flow/kitti15-main-flow.pth.tar', SINTEL, ['--flow-root', str(SINTEL_FLOW)]),
    'VI-FT3D-KITTI-FLOW': ('flow', 'flyingthings3d', 'checkpoints/main-results/flow/kitti15-main-flow.pth.tar', FLYINGTHINGS3D, ['--allow-candidate']),
    'VI-FT3D-KITTI-DISP': ('depth', 'flyingthings3d', 'checkpoints/main-results/flow/kitti15-main-flow.pth.tar', FLYINGTHINGS3D, ['--allow-candidate']),
    'VI-FT3D-ADAPTED-FLOW': ('flow', 'flyingthings3d', 'checkpoints/main-results/flow/flyingthings3d-spring-transfer-flow.pth.tar', FLYINGTHINGS3D, ['--allow-candidate']),
    'VI-FT3D-ADAPTED-DISP': ('depth', 'flyingthings3d', 'checkpoints/main-results/flow/flyingthings3d-spring-transfer-flow.pth.tar', FLYINGTHINGS3D, ['--allow-candidate']),
    'VI-SPRING-KITTI-FLOW': ('flow', 'spring', 'checkpoints/main-results/flow/kitti15-main-flow.pth.tar', SPRING_FULL, ['--allow-candidate', '--full', '--model-implementation', 'historical-final']),
    'VI-SPRING-ADAPTED-FLOW': ('flow', 'spring', 'checkpoints/main-results/flow/flyingthings3d-spring-transfer-flow.pth.tar', SPRING_ADAPTED, ['--allow-candidate', '--model-implementation', 'historical-final']),
    'VI-SPRING-ADAPTED-DISP': ('depth', 'spring', 'checkpoints/main-results/flow/flyingthings3d-spring-transfer-flow.pth.tar', SPRING_ADAPTED, ['--allow-candidate', '--model-implementation', 'historical-final']),
    'IX': ('depth', 'kitti2015', 'checkpoints/main-results/depth/kitti15-main-depth.pth.tar', KITTI_2015, []),
}

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    p = argparse.ArgumentParser()
    p.add_argument('--table', choices=EXPERIMENTS, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--root', type=Path, help='Override the configured dataset root.')
    p.add_argument('--limit', type=int, help='Evaluate an explicit prefix of the mapped dataset.')
    p.add_argument('--full', action='store_true', help='Evaluate the complete dataset at --root.')
    a = p.parse_args()
    task, dataset, checkpoint, root, extra = EXPERIMENTS[a.table]
    root = a.root or root
    if not root or not Path(root).is_dir():
        raise SystemExit(f'Missing dataset root for Table {a.table}: {root}')
    out = ROOT / 'outputs' / f'table_{a.table}_{datetime.now():%Y%m%d_%H%M%S}'
    cmd = [sys.executable, '-u', str(ROOT/'core'/f'evaluate_{task}.py'), '--dataset', dataset,
           '--root', str(root), '--checkpoint', str(ROOT/checkpoint), '--device', a.device,
           '--output', str(out), *extra]
    if a.limit is not None:
        cmd += ['--limit', str(a.limit)]
    if a.full:
        cmd += ['--full']
    child = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, text=True,
                           env=dict(os.environ, PYTHONUNBUFFERED='1'))
    code = child.returncode
    if code:
        if child.stdout:
            print(child.stdout, end='', flush=True)
        raise SystemExit(code)
    summary_path = out / 'summary.json'
    if not summary_path.is_file():
        print(f'\nEvaluation completed without {summary_path}; refusing to print stale results.', flush=True)
        raise SystemExit(1)
    summary = json.loads(summary_path.read_text())
    metrics = dict(summary)
    if task == 'flow':
        metrics = {
            ('f1' if name == 'f1_5pct' else name): value
            for name, value in metrics.items()
        }
    lines = [
        'CALCULATED RESULTS',
        '=========================================',
        *[f'{name} : {value}' for name, value in metrics.items()],
        '=========================================',
    ]
    report = '\n'.join(lines) + '\n'
    sys.stdout.flush()
    os.write(sys.stdout.fileno(), ('\n' + report).encode())
    sys.stdout.flush()
    (out/'terminal_metrics.txt').write_text(report)

if __name__ == '__main__': main()
