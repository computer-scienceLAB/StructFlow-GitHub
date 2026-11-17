import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from inference.core import Predictor,PROFILES,calibration,fresh_output,write_metadata
from inference.visualization import flow_rgb,scalar_rgb,labeled_panel,save_legend


def main(task):
    p=argparse.ArgumentParser(description='Synchronized StructFlow video from consecutive stereo frames.')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True,help='Directory of left images from ONE sequence.')
    p.add_argument('--right-input',type=Path,required=True,help='Right images with corresponding filenames.')
    p.add_argument('--output',type=Path,required=True,help='New output directory containing MP4, arrays, metadata, legend.')
    p.add_argument('--device',default='cpu')
    p.add_argument('--profile',choices=PROFILES,default='kitti-original')
    p.add_argument('--dataset',required=True)
    p.add_argument('--ffmpeg',default='ffmpeg',help='ffmpeg executable name or explicit path (must support libx264).')
    p.add_argument('--pattern',default='*.png')
    p.add_argument('--start',type=int,default=0,help='Index in numerically sorted frame list.')
    p.add_argument('--frames',type=int,default=60,help='Number of prediction frames; requires one extra image.')
    p.add_argument('--fps',type=float,default=10,help='Playback rate; recorded, no interpolated frames.')
    p.add_argument('--visual-max',type=float,default=256 if task=='flow' else 80)
    if task=='depth':
        p.add_argument('--quantity',choices=['depth','disparity'],default='depth')
        p.add_argument('--calibration',type=Path)
        p.add_argument('--focal-px',type=float)
        p.add_argument('--baseline-m',type=float)
    a=p.parse_args()
    encoder=shutil.which(a.ffmpeg)
    if a.ffmpeg=='ffmpeg':
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            encoder=get_ffmpeg_exe()
        except ImportError:
            pass
    if encoder is None: p.error('ffmpeg not found; install it or pass --ffmpeg /path/to/ffmpeg.')
    encoders=subprocess.check_output([encoder,'-hide_banner','-encoders'],stderr=subprocess.STDOUT,text=True)
    if 'libx264' not in encoders: p.error('Selected ffmpeg lacks libx264; install requirements-video.txt or supply another --ffmpeg.')
    if a.frames<1 or a.start<0 or a.fps<=0: p.error('Positive --frames/--fps and nonnegative --start required.')
    def index(path):
        match=re.search(r'(\d+)$',path.stem)
        if not match: raise ValueError('Frame filenames must end in numeric indices.')
        return int(match[1])
    frames=sorted(a.input.glob(a.pattern),key=index)[a.start:a.start+a.frames+1]
    if len(frames)!=a.frames+1: p.error('Insufficient sequential frames for requested clip.')
    if any(index(b)!=index(c)+1 for c,b in zip(frames,frames[1:])): p.error('Frame numbering has gaps; refusing temporal interpolation.')
    right=[a.right_input/f.name for f in frames]
    if not all(f.is_file() for f in right): p.error('Missing corresponding right-camera frames.')
    quantity='flow' if task=='flow' else a.quantity
    fb=None
    if quantity=='depth':
        if a.calibration: f,b=calibration(a.calibration)
        elif a.focal_px and a.baseline_m and a.focal_px>0 and a.baseline_m>0: f,b=a.focal_px,a.baseline_m
        else: p.error('Metric depth needs --calibration or positive --focal-px/--baseline-m.')
        fb=f*b
    model=Predictor(a.checkpoint,a.device,a.profile)
    out=fresh_output(a.output); arrays=out/'predictions'; arrays.mkdir()
    proc=None
    try:
        for i in range(a.frames):
            paths=[frames[i],frames[i+1],right[i],right[i+1]]
            pred=model.predict(paths)
            value=pred['flow'] if task=='flow' else pred['disparity']
            if fb is not None: value=fb/np.maximum(value,1e-3)
            np.save(arrays/f'{i:06d}_{quantity}.npy',value.astype(np.float32))
            rgb=np.asarray(Image.open(frames[i]).convert('RGB'))
            vis=flow_rgb(value,a.visual_max) if task=='flow' else scalar_rgb(value,a.visual_max)
            unit='m' if quantity=='depth' else 'px'
            panel=labeled_panel([rgb,vis],[f'{a.dataset} | frame {index(frames[i])}',f'StructFlow {quantity} | scale 0-{a.visual_max:g} {unit}'])
            width=panel.width+(panel.width%2); height=panel.height+(panel.height%2)
            canvas=Image.new('RGB',(width,height)); canvas.paste(panel)
            if proc is None:
                cmd=[encoder,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(a.fps),'-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{quantity}.mp4')]
                proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
            proc.stdin.write(np.asarray(canvas).tobytes())
        proc.stdin.close()
        if proc.wait()!=0: raise RuntimeError('ffmpeg encoding failed')
    finally:
        if proc is not None and proc.poll() is None:
            proc.stdin.close(); proc.wait()
    meta=dict(dataset=a.dataset,task=quantity,checkpoint=a.checkpoint.name,
              units='metres' if quantity=='depth' else 'pixels',fps=a.fps,frames=a.frames)
    write_metadata(out/'metadata.json',meta)
    save_legend(out/'legend.png','flow' if task=='flow' else quantity+' ('+unit+')',a.visual_max)
    print(out)
