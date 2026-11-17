"""Scientific flow/disparity formats. PNG decoding preserves 16-bit channels."""
import struct
from pathlib import Path
import cv2
import numpy as np


def read_flow(path):
    path = Path(path)
    if path.suffix == '.png':
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None or im.dtype != np.uint16 or im.ndim != 3:
            raise ValueError(f'Expected KITTI uint16 3-channel flow: {path}')
        return (im[..., 2:0:-1].astype(np.float32) - 32768) / 64, im[..., 0] > 0
    if path.suffix == '.flo':
        with path.open('rb') as f:
            magic, w, h = struct.unpack('<fii', f.read(12))
            if magic != 202021.25 or min(w, h) <= 0:
                raise ValueError(f'Invalid .flo header: {path}')
            a = np.frombuffer(f.read(), dtype='<f4').reshape(h, w, 2).copy()
        return a, np.isfinite(a).all(-1) & (np.abs(a) < 1e9).all(-1)
    if path.suffix == '.pfm':
        a = read_pfm(path)
        valid = np.isfinite(a[..., :2]).all(-1)
        if a.shape[-1] >= 3 and np.nanmax(a[..., 2]) > 0:
            valid &= a[..., 2] > .5
        return a[..., :2], valid
    if path.suffix == '.flo5':
        import h5py
        with h5py.File(path) as f:
            a = np.asarray(f['flow'], dtype=np.float32)
        return a, np.isfinite(a).all(-1)
    raise ValueError(f'Unsupported flow format: {path.suffix}')


def read_pfm(path):
    with open(path, 'rb') as f:
        head = f.readline().decode().strip()
        if head not in ('PF', 'Pf'):
            raise ValueError('Invalid PFM header')
        dims = f.readline().decode().strip()
        while dims.startswith('#'):
            dims = f.readline().decode().strip()
        w, h = map(int, dims.split())
        scale = float(f.readline())
        a = np.frombuffer(f.read(), dtype='<f4' if scale < 0 else '>f4')
    return np.flipud(a.reshape((h, w, 3) if head == 'PF' else (h, w))).astype(np.float32).copy()


def read_disparity(path):
    path = Path(path)
    if path.suffix == '.png':
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None or a.ndim != 2 or a.dtype != np.uint16:
            raise ValueError('Expected KITTI uint16 disparity PNG')
        a = a.astype(np.float32) / 256
    elif path.suffix == '.pfm':
        a = read_pfm(path)
    elif path.suffix == '.dsp5':
        import h5py
        with h5py.File(path) as f:
            a = np.asarray(f['disparity'], dtype=np.float32)
        return a, np.isfinite(a)
    else:
        raise ValueError(f'Unsupported disparity format: {path.suffix}')
    return a, np.isfinite(a) & (a > 0)


def write_flow(path, flow):
    path = Path(path)
    h, w = flow.shape[:2]
    if path.suffix == '.flo':
        with path.open('wb') as f:
            f.write(struct.pack('<fii', 202021.25, w, h))
            f.write(flow.astype('<f4').tobytes())
    elif path.suffix == '.png':
        encoded = flow * 64 + 32768
        if np.any(encoded < 0) or np.any(encoded > 65535):
            raise ValueError('Flow exceeds KITTI PNG representable range; use .flo or .npy.')
        image = np.ones((h, w, 3), dtype=np.uint16)
        image[..., 2:0:-1] = np.rint(encoded).astype(np.uint16)
        if not cv2.imwrite(str(path), image):
            raise IOError(path)
    else:
        raise ValueError('Expected .flo or .png')
