#!/usr/bin/env python3
"""Recompute examples from a portable manifest and real checkpoint inference."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from inference.core import Predictor,PROFILES,calibration,fresh_output,write_metadata
from inference.formats import read_flow,read_disparity
from inference.visualization import flow_rgb,scalar_rgb,labeled_panel,save_legend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--task',choices=['flow','depth','disparity'],required=True)
    p.add_argument('--profile',choices=PROFILES,default='kitti-original')
    p.add_argument('--visual-max',type=float)
    a=p.parse_args()
    if a.visual_max is None:
        a.visual_max = 256 if a.task == 'flow' else 80
    manifest=json.loads(a.manifest.read_text())
    model=Predictor(a.checkpoint,a.device,a.profile); out=fresh_output(a.output)
    for row in manifest['samples']:
        paths=[a.root/x for x in row['inputs']]
        pred=model.predict(paths); directory=out/row['id'].replace('/','_').replace('|','_'); directory.mkdir()
        rgb=np.asarray(Image.open(paths[0]).convert('RGB'))
        images=[rgb]; labels=['Reference RGB']
        render=lambda x,valid=None: flow_rgb(x,a.visual_max) if a.task=='flow' else scalar_rgb(x,a.visual_max,valid)
        key='flow' if a.task=='flow' else 'disparity'
        value=pred[key]
        gt=None
        if key+'_gt' in row:
            gt,valid=(read_flow if key=='flow' else read_disparity)(a.root/row[key+'_gt'])
        if a.task=='depth':
            if 'calibration' not in row: raise ValueError('Metric depth requires a calibration path per manifest row.')
            f,b=calibration(a.root/row['calibration']); value=f*b/np.maximum(value,1e-3)
            if gt is not None: gt=f*b/np.maximum(gt,1e-3)
        if gt is not None:
            colored=render(gt,valid); colored[~valid]=0
            images.append(colored); labels.append('Ground truth')
        images.append(render(value)); labels.append('StructFlow '+a.task)
        panel=labeled_panel(images,labels); panel.save(directory/'comparison.png')
        np.save(directory/f'{a.task}.npy',value)
        Image.fromarray(render(value)).save(directory/f'{a.task}_visualization.png')
        meta=model.metadata(paths); meta.update(sample_id=row['id'],dataset=manifest['dataset'],task=a.task,
            figure_status=row.get('figure_status','illustrative'),visual_max=a.visual_max)
        write_metadata(directory/'metadata.json',meta)
    save_legend(out/'legend.png','flow' if a.task=='flow' else a.task+(' (m)' if a.task=='depth' else ' (px)'),a.visual_max)


if __name__=='__main__': main()
