"""Inference with the original StructFlow architecture and explicit protocols."""
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import resize

from models.pwcnet import PWCFlowDepth

ROOT = Path(__file__).resolve().parents[2]

PROFILES = {
    'kitti-original': {'size': (256, 832), 'input_resize': 'skimage', 'scalar_resize': 'skimage', 'flow_resize': 'opencv'},
    'eigen-recheck': {'size': (256, 832), 'input_resize': 'opencv', 'scalar_resize': 'opencv', 'flow_resize': 'opencv'},
    'synthetic-original': {'size': (256, 832), 'input_resize': 'pillow', 'scalar_resize': 'opencv', 'flow_resize': 'opencv'},
    'sintel-original': {'size': (384, 768), 'input_resize': 'skimage', 'scalar_resize': 'opencv', 'flow_resize': 'torch-align-corners'},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fresh_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def calibration(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if ':' in line:
            key, val = line.split(':', 1)
            try:
                values[key] = np.array([float(x) for x in val.split()])
            except ValueError:
                pass
    p2 = values.get('P_rect_02', values.get('P2'))
    p3 = values.get('P_rect_03', values.get('P3'))
    if p2 is None or p3 is None:
        raise ValueError('Calibration requires P_rect_02/P_rect_03 or P2/P3.')
    p2, p3 = p2.reshape(3, 4), p3.reshape(3, 4)
    focal = float(p2[0, 0])
    baseline = float(p2[0, 3] / p2[0, 0] - p3[0, 3] / p3[0, 0])
    if focal <= 0 or baseline <= 0:
        raise ValueError('Expected positive focal length and rectified left-to-right baseline.')
    return focal, baseline


def scalar_resize(x, hw, method):
    if method == 'skimage':
        return resize(x, hw).astype(np.float32)
    return cv2.resize(x, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)


def resize_flow(x, hw, method='opencv'):
    h, w = x.shape[:2]
    if method == 'torch-align-corners':
        y = F.interpolate(torch.from_numpy(x.transpose(2, 0, 1).copy())[None], hw,
                          mode='bilinear', align_corners=True)[0].numpy().transpose(1, 2, 0).copy()
        y *= np.array([hw[1] / w, hw[0] / h], dtype=np.float32)
        return y
    # Preserve the original KITTI evaluator's scale-before-interpolate order.
    y = x.copy()
    y[..., 0] = y[..., 0] / w * hw[1]
    y[..., 1] = y[..., 1] / h * hw[0]
    return cv2.resize(y, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)


class Predictor:
    def __init__(self, checkpoint, device='cpu', profile='kitti-original', size=None,
                 model_implementation='release'):
        self.checkpoint = Path(checkpoint)
        self.checksum = sha256(checkpoint)
        self.device = torch.device(device)
        self.profile = profile
        self.settings = PROFILES[profile]
        self.size = tuple(size or self.settings['size'])
        if any(d < 64 or d % 64 for d in self.size):
            raise ValueError('Input height and width must be positive multiples of 64 (minimum 64).')
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = ck.get('state_dict', ck.get('model_state_dict', ck))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        if model_implementation == 'release':
            model_class = PWCFlowDepth
        elif model_implementation == 'historical-final':
            original_root = Path(os.environ.get(
                'STRUCTFLOW_ORIGINAL_ROOT', ROOT.parent / 'StructFlow_code')).expanduser()
            if not (original_root / 'tools/reviewer2_q5/final_structflow_imports.py').is_file():
                raise FileNotFoundError(
                    'Historical Spring model source is unavailable; set STRUCTFLOW_ORIGINAL_ROOT.')
            sys.path.insert(0, str(original_root))
            from tools.reviewer2_q5.final_structflow_imports import load_final_structflow_classes
            model_class, _, _ = load_final_structflow_classes()
        else:
            raise ValueError(f'Unknown model implementation: {model_implementation}')
        self.model = model_class()
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.epoch = ck.get('epoch')

    def tensors(self, paths):
        if len(paths) != 4:
            raise ValueError('Supply left_t left_t1 right_t right_t1, in this order.')
        images = [np.asarray(Image.open(p).convert('RGB')) for p in paths]
        if len({im.shape for im in images}) != 1:
            raise ValueError('All four rectified images must have the same dimensions.')
        tensors = []
        h, w = self.size
        for image in images:
            method = self.settings['input_resize']
            if method == 'pillow':
                x = np.asarray(Image.fromarray(image).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
            elif method == 'skimage':
                x = resize(image.astype(np.float32), (h, w))
            else:
                x = cv2.resize(image.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            tensors.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float() / 255.0)
        return torch.cat(tensors, 0)[None].to(self.device), images[0].shape[:2]

    @torch.inference_mode()
    def predict(self, paths, output_shape=None):
        inp, native_shape = self.tensors(paths)
        flows, disparities = self.model(inp, inp[:, :3], with_bk=True)
        raw_flow = flows[0][0, :2].cpu().numpy().transpose(1, 2, 0).copy()
        normalized_disp = disparities[0][0, 0].cpu().numpy().copy()
        hw = tuple(output_shape or native_shape)
        flow = resize_flow(raw_flow, hw, self.settings['flow_resize'])
        disp = scalar_resize(normalized_disp, hw, self.settings['scalar_resize']) * hw[1]
        if not np.isfinite(flow).all() or not np.isfinite(disp).all():
            raise ValueError('Non-finite prediction: evaluation aborted, no invalid pixels silently discarded.')
        return dict(flow=flow, disparity=disp.astype(np.float32), raw_flow=raw_flow,
                    normalized_disparity=normalized_disp, native_shape=native_shape)

    def metadata(self, paths):
        return dict(checkpoint=self.checkpoint.name, inputs=[Path(p).name for p in paths])


def write_metadata(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
