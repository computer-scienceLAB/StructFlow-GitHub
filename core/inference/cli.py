import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .core import Predictor, PROFILES, calibration, fresh_output, write_metadata
from .formats import write_flow
from .visualization import flow_rgb, scalar_rgb, save_legend


def parser(task):
    p=argparse.ArgumentParser(description=f'StructFlow {task} inference (four rectified RGB images).')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input',type=Path,nargs=4,required=True,metavar=('LEFT_T','LEFT_T1','RIGHT_T','RIGHT_T1'))
    p.add_argument('--output',type=Path,required=True,help='New output directory; existing directories are rejected.')
    p.add_argument('--device',default='cpu')
    p.add_argument('--profile',choices=PROFILES,default='kitti-original')
    p.add_argument('--size',nargs=2,type=int,metavar=('HEIGHT','WIDTH'))
    p.add_argument('--visualize',action='store_true')
    p.add_argument('--visual-max',type=float,default=256.0 if task=='flow' else 80.0)
    if task=='flow':
        p.add_argument('--format',choices=['flo','kitti-png'],default='flo')
    else:
        p.add_argument('--quantity',choices=['depth','disparity'],default='depth')
        p.add_argument('--calibration',type=Path,help='KITTI rectified calibration at original input resolution.')
        p.add_argument('--focal-px',type=float)
        p.add_argument('--baseline-m',type=float)
    return p


def main(task):
    p=parser(task); args=p.parse_args()
    fb=None
    if task=='depth' and args.quantity=='depth':
        if args.calibration:
            f,b=calibration(args.calibration)
        elif args.focal_px and args.baseline_m and args.focal_px>0 and args.baseline_m>0:
            f,b=args.focal_px,args.baseline_m
        else:
            p.error('Metric depth requires --calibration or positive --focal-px and --baseline-m. Use --quantity disparity otherwise.')
        fb=f*b
    model=Predictor(args.checkpoint,args.device,args.profile,args.size)
    pred=model.predict(args.input)
    out=fresh_output(args.output)
    meta=model.metadata(args.input)
    if task=='flow':
        value=pred['flow']; np.save(out/'flow.npy',value)
        write_flow(out/('flow.flo' if args.format=='flo' else 'flow_kitti.png'),value)
        meta.update(task='optical flow',units='pixels in original image coordinates',shape=list(value.shape))
        if args.visualize:
            Image.fromarray(flow_rgb(value,args.visual_max)).save(out/'flow_visualization.png')
            save_legend(out/'flow_legend.png','flow',args.visual_max)
    else:
        value=pred['disparity'] if fb is None else fb/np.maximum(pred['disparity'],1e-3)
        np.save(out/f'{args.quantity}.npy',value.astype(np.float32))
        meta.update(task=args.quantity,units='pixels' if fb is None else 'metres',focal_times_baseline=fb,
                    shape=list(value.shape),conversion='d_px = normalized_disparity * original_width; z = f*b / max(d_px, 0.001)')
        if args.visualize:
            Image.fromarray(scalar_rgb(value,args.visual_max)).save(out/f'{args.quantity}_visualization.png')
            save_legend(out/f'{args.quantity}_legend.png',args.quantity+' ('+meta['units']+')',args.visual_max)
    meta['visualization_maximum']=args.visual_max if args.visualize else None
    write_metadata(out/'metadata.json',meta)
    print(out)
