#!/usr/bin/env python3
"""Run StructFlow on a user-selected rectified stereo sequence and render final predictions."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
from inference.core import Predictor, PROFILES
from inference.visualization import flow_rgb, scalar_rgb


DATASET_PROFILES = {
    'kitti2012': 'kitti-original', 'kitti2015': 'kitti-original',
    'sintel': 'sintel-original', 'flyingthings3d': 'synthetic-original',
    'spring': 'synthetic-original',
}
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp'}


def images(directory):
    result = sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if len(result) < 2:
        raise ValueError(f'{directory} must contain at least two image files.')
    return result


def resolve_inputs(left, right, dataset):
    left = Path(left)
    if right is None and (left / 'left').is_dir() and (left / 'right').is_dir():
        left, right = left / 'left', left / 'right'
    if right is None:
        raise ValueError('--input-right is required unless --input contains left/ and right/ subdirectories.')
    left_images, right_images = images(left), images(right)
    if len(left_images) != len(right_images):
        raise ValueError('Left and right image folders must contain the same number of ordered frames.')
    if dataset == 'spring':
        left_ids = [p.stem.rsplit('_', 1)[-1] for p in left_images]
        right_ids = [p.stem.rsplit('_', 1)[-1] for p in right_images]
        if left_ids != right_ids:
            raise ValueError('Spring left/right folders must contain matching ordered frame IDs.')
    elif [p.name for p in left_images] != [p.name for p in right_images]:
        raise ValueError('Left and right image filenames must match exactly for deterministic stereo pairing.')
    return left_images, right_images


def pairs(dataset, left, right):
    if dataset in ('kitti2012', 'kitti2015'):
        lookup = {p.name: p for p in left}
        right_lookup = {p.name: p for p in right}
        result = []
        for name in sorted(lookup):
            if not name.endswith('_10.png'):
                continue
            nxt = name[:-7] + '_11.png'
            if nxt in lookup and name in right_lookup and nxt in right_lookup:
                result.append((lookup[name], lookup[nxt], right_lookup[name], right_lookup[nxt]))
        if result:
            return result

        # Pair only adjacent frames from a single KITTI Raw camera stream.
        indexed = []
        for image in left:
            if image.stem.isdigit() and image.name in right_lookup:
                indexed.append((int(image.stem), image))
        indexed.sort()
        result = [
            (first, second, right_lookup[first.name], right_lookup[second.name])
            for (first_id, first), (second_id, second) in zip(indexed, indexed[1:])
            if second_id == first_id + 1
        ]
        if result:
            return result
        raise ValueError(
            'KITTI input folders require matching <scene>_10.png/<scene>_11.png '
            'benchmark pairs or consecutive numeric KITTI Raw filenames.'
        )
    return [(left[i], left[i + 1], right[i], right[i + 1]) for i in range(len(left) - 1)]


def panel(rgb, visualization, task):
    left = Image.fromarray(rgb).convert('RGB')
    right = Image.fromarray(visualization).resize(left.size, Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (left.width * 2, left.height + 34), 'black')
    canvas.paste(left, (0, 34)); canvas.paste(right, (left.width, 34))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError: font = ImageFont.load_default()
    draw.text((8, 8), 'RGB', fill='white', font=font)
    draw.text((left.width + 8, 8), 'StructFlow ' + ('flow' if task == 'flow' else 'disparity'), fill='white', font=font)
    return canvas


def combined_panel(rgb, flow, disparity):
    images = [Image.fromarray(rgb).convert('RGB')]
    images.append(Image.fromarray(flow).resize(images[0].size, Image.Resampling.BILINEAR))
    images.append(Image.fromarray(disparity).resize(images[0].size, Image.Resampling.BILINEAR))
    width, height = images[0].size
    canvas = Image.new('RGB', (width * 3, height + 34), 'black')
    for index, image in enumerate(images):
        canvas.paste(image, (index * width, 34))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError: font = ImageFont.load_default()
    for index, title in enumerate(('Original', 'Flow', 'Disparity')):
        draw.text((index * width + 8, 8), title, fill='white', font=font)
    return canvas


def write_video(frames, path, fps):
    if not frames:
        return
    size = frames[0].size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise RuntimeError('Could not create MP4 video with OpenCV.')
    for frame in frames:
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=DATASET_PROFILES)
    parser.add_argument('--task', required=True, choices=['flow', 'disp', 'disparity', 'combined'])
    parser.add_argument('--input', required=True, help='Left-frame folder, or a folder containing left/ and right/.')
    parser.add_argument('--input-right', help='Right-frame folder with matching ordered filenames.')
    parser.add_argument('--checkpoint', type=Path,
                        help='Checkpoint for flow or disparity visualization.')
    parser.add_argument('--flow-checkpoint', type=Path,
                        help='Optical-flow checkpoint for --task combined.')
    parser.add_argument('--disp-checkpoint', type=Path,
                        help='Disparity checkpoint for --task combined.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow replacing generated files in an existing output directory.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--visual-max', type=float, default=80.0)
    parser.add_argument('--flow-max', type=float, default=256.0,
                        help='Original HSV flow visualization scale in pixels.')
    parser.add_argument('--max-pairs', type=int, help='Optional deterministic prefix for a short preview.')
    args = parser.parse_args()
    if args.task == 'combined':
        if not args.flow_checkpoint or not args.flow_checkpoint.is_file():
            parser.error('--flow-checkpoint must name an existing checkpoint for --task combined.')
        if not args.disp_checkpoint or not args.disp_checkpoint.is_file():
            parser.error('--disp-checkpoint must name an existing checkpoint for --task combined.')
    elif not args.checkpoint or not args.checkpoint.is_file():
        parser.error('--checkpoint must name an existing checkpoint.')
    if args.output.exists() and not args.overwrite:
        parser.error(f'Output already exists: {args.output}')
    left, right = resolve_inputs(args.input, args.input_right, args.dataset)
    selected_pairs = pairs(args.dataset, left, right)
    if args.max_pairs is not None:
        if args.max_pairs <= 0:
            parser.error('--max-pairs must be positive.')
        selected_pairs = selected_pairs[:args.max_pairs]
    args.output.mkdir(parents=True, exist_ok=args.overwrite)
    task = 'flow' if args.task == 'flow' else 'disp'
    rendered = []
    if args.task == 'combined':
        flow_model = Predictor(args.flow_checkpoint, args.device, DATASET_PROFILES[args.dataset])
        disp_model = Predictor(args.disp_checkpoint, args.device, DATASET_PROFILES[args.dataset])
        directory = args.output / 'combined'
    else:
        model = Predictor(args.checkpoint, args.device, DATASET_PROFILES[args.dataset])
        directory = args.output / ('flow' if task == 'flow' else 'disp')
    directory.mkdir(exist_ok=args.overwrite)
    for index, paths in enumerate(selected_pairs):
        if args.task == 'combined':
            flow = flow_rgb(flow_model.predict(paths)['flow'], args.flow_max)
            disparity = scalar_rgb(disp_model.predict(paths)['disparity'], args.visual_max)
            rendered_frame = combined_panel(np.asarray(Image.open(paths[0]).convert('RGB')), flow, disparity)
            rendered_frame.save(directory / f'{index:06d}.png')
            if args.video:
                rendered.append(rendered_frame)
        else:
            pred = model.predict(paths)
            value = pred['flow'] if task == 'flow' else pred['disparity']
            rgb = flow_rgb(value, args.flow_max) if task == 'flow' else scalar_rgb(value, args.visual_max)
            Image.fromarray(rgb).save(directory / f'{index:06d}.png')
            if args.video:
                rendered.append(panel(np.asarray(Image.open(paths[0]).convert('RGB')), rgb, task))
    if args.video:
        name = 'original_flow_disparity_video.mp4' if args.task == 'combined' else ('flow_video.mp4' if task == 'flow' else 'disp_video.mp4')
        write_video(rendered, args.output / name, args.fps)


if __name__ == '__main__':
    main()
