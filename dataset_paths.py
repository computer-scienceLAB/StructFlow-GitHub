"""Portable dataset configuration with optional local overrides.

Set ``STRUCTFLOW_DATA_ROOT`` or task-specific environment variables on any
machine. A git-ignored ``dataset_paths_local.py`` may define the same names for
one local installation; dataset data are not included in this repository.
"""
import os
from pathlib import Path


DATA_ROOT = Path(os.environ.get('STRUCTFLOW_DATA_ROOT', 'data')).expanduser()


def configured(name, relative):
    return Path(os.environ.get(name, str(DATA_ROOT / relative))).expanduser()


KITTI_2015 = configured('STRUCTFLOW_KITTI_2015', 'kitti2015/training')
KITTI_2012 = configured('STRUCTFLOW_KITTI_2012', 'kitti2012/training')
KITTI_RAW = configured('STRUCTFLOW_KITTI_RAW', 'kitti_raw')
SINTEL = configured('STRUCTFLOW_SINTEL', 'sintel/training')
SINTEL_FLOW = configured('STRUCTFLOW_SINTEL_FLOW', 'sintel/training/flow')
FLYINGTHINGS3D = configured('STRUCTFLOW_FLYINGTHINGS3D', 'flyingthings3d/tableVI_500')
SPRING_FULL = Path(os.environ.get('SPRING_DATA_ROOT', os.environ.get('STRUCTFLOW_SPRING', str(DATA_ROOT / 'spring/train')))).expanduser()
SPRING_ADAPTED = Path(os.environ.get('SPRING_ADAPTED_DATA_ROOT', str(DATA_ROOT / 'spring/tableVI_200/train'))).expanduser()
SPRING = SPRING_FULL

try:
    from dataset_paths_local import (  # type: ignore[import-not-found]
        FLYINGTHINGS3D as _FLYINGTHINGS3D,
        KITTI_2012 as _KITTI_2012,
        KITTI_2015 as _KITTI_2015,
        KITTI_RAW as _KITTI_RAW,
        SINTEL as _SINTEL,
        SINTEL_FLOW as _SINTEL_FLOW,
        SPRING_FULL as _SPRING_FULL,
        SPRING_ADAPTED as _SPRING_ADAPTED,
    )
except ImportError:
    pass
else:
    KITTI_2015 = Path(_KITTI_2015)
    KITTI_2012 = Path(_KITTI_2012)
    KITTI_RAW = Path(_KITTI_RAW)
    SINTEL = Path(_SINTEL)
    SINTEL_FLOW = Path(_SINTEL_FLOW)
    FLYINGTHINGS3D = Path(_FLYINGTHINGS3D)
    SPRING_FULL = Path(os.environ.get('SPRING_DATA_ROOT', _SPRING_FULL)).expanduser()
    SPRING_ADAPTED = Path(os.environ.get('SPRING_ADAPTED_DATA_ROOT', _SPRING_ADAPTED)).expanduser()
    SPRING = SPRING_FULL
