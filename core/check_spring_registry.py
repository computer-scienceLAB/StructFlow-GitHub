#!/usr/bin/env python3
"""Validate that Spring table commands remain bound to their intended data."""
from pathlib import Path

from evaluate import EXPERIMENTS
from evaluation.datasets import samples


def ready_spring_pairs(root: Path, task: str) -> int:
    key = 'flow' if task == 'flow' else 'disparity'
    return sum(all(path.is_file() for path in [*row['inputs'], row[key]])
               for row in samples('spring', root))


def main() -> None:
    full = EXPERIMENTS['VI-SPRING-KITTI-FLOW']
    adapted = EXPERIMENTS['VI-SPRING-ADAPTED-DISP']
    if full[0:3] != ('flow', 'spring', 'checkpoints/main-results/flow/kitti15-main-flow.pth.tar'):
        raise SystemExit('KITTI-only Spring flow registry binding changed.')
    if adapted[0:3] != ('depth', 'spring', 'checkpoints/main-results/flow/flyingthings3d-spring-transfer-flow.pth.tar'):
        raise SystemExit('Adapted Spring disparity registry binding changed.')
    if 'historical-final' not in full[4] or 'historical-final' not in adapted[4]:
        raise SystemExit('Spring commands must use the recovered historical model implementation.')
    if ready_spring_pairs(Path(adapted[3]), adapted[0]) != 200:
        raise SystemExit('Adapted Spring disparity root is not the retained 200-pair evaluation subset.')
    full_ready = ready_spring_pairs(Path(full[3]), full[0])
    if full_ready != 4963:
        raise SystemExit(f'Full Spring flow extraction is incomplete: expected 4963 ready pairs, found {full_ready}.')
    print('Spring registry verified.')


if __name__ == '__main__':
    main()
