"""Run with python -m evaluation.evaluate."""
import argparse
import json
from pathlib import Path
import numpy as np
from .datasets import samples,check_files,eigen_gt
from .metrics import flow_metrics,disparity_metrics,depth_metrics
from inference.core import Predictor,PROFILES,calibration,fresh_output,write_metadata
from inference.formats import read_flow,read_disparity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['kitti2015','kitti2012','eigen','flyingthings3d','spring','sintel'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--split',type=Path)
    p.add_argument('--flow-root',type=Path)
    p.add_argument('--profile',choices=PROFILES)
    p.add_argument('--size',nargs=2,type=int)
    p.add_argument('--task',choices=['flow','depth','joint'],default='joint')
    p.add_argument('--model-implementation', choices=['release', 'historical-final'], default='release')
    p.add_argument('--limit',type=int,help='Explicit prefix subset.')
    p.add_argument('--save-predictions',action='store_true')
    p.add_argument('--detailed-records',action='store_true')
    a=p.parse_args()
    profile=a.profile or {'eigen':'eigen-recheck','sintel':'sintel-original','flyingthings3d':'synthetic-original','spring':'synthetic-original'}.get(a.dataset,'kitti-original')
    rows=samples(a.dataset,a.root,a.split,a.flow_root); total=len(rows)
    if a.limit is not None:
        if a.limit<=0: p.error('--limit must be positive')
        rows=rows[:a.limit]
    checked_rows=[{k:v for k,v in r.items() if not (a.task=='depth' and k in ('flow','noc')) and not (a.task=='flow' and k in ('disparity','calibration','velodyne','velo_calibration'))} for r in rows]
    check_files(checked_rows)
    if a.dataset=='eigen' and a.task=='flow': p.error('Eigen split has no flow GT in this evaluator.')
    if a.dataset in ('sintel','kitti2012') and a.task=='depth': p.error('Use --task flow; no depth GT configured for this dataset.')
    dataset_root = {'kitti2015': 'KITTI2015/training', 'kitti2012': 'KITTI2012/training',
                    'eigen': 'KITTI-Raw', 'flyingthings3d': 'FlyingThings3D',
                    'spring': 'Spring/train', 'sintel': 'Sintel/training'}[a.dataset]

    def public_path(path):
        try:
            return f'{dataset_root}/{Path(path).relative_to(a.root).as_posix()}'
        except ValueError:
            return f'{dataset_root}/{Path(path).name}'

    model=Predictor(a.checkpoint,a.device,profile,a.size,a.model_implementation)
    out=fresh_output(a.output); results=[]; manifest=[]
    for i,row in enumerate(rows):
        gt_flow=valid=noc=gt_disp=dvalid=None
        if 'flow' in row and a.task!='depth':
            gt_flow,valid=read_flow(row['flow'])
            if 'noc' in row: _,noc=read_flow(row['noc'])
        if a.task!='flow' and 'disparity' in row:
            gt_disp,dvalid=read_disparity(row['disparity'])
        output_shape = gt_flow.shape[:2] if gt_flow is not None else (gt_disp.shape[:2] if gt_disp is not None else None)
        pred=model.predict(row['inputs'],output_shape)
        metrics={'sample_id':row['id']}
        if a.task!='depth' and gt_flow is not None: metrics.update(flow_metrics(pred['flow'],gt_flow,valid,noc))
        if a.task!='flow':
            if gt_disp is not None:
                metrics.update(disparity_metrics(pred['disparity'],gt_disp,dvalid))
                if 'calibration' in row:
                    f,b=calibration(row['calibration'])
                    gt_depth=f*b/np.maximum(gt_disp,1e-3)
                    metrics.update(depth_metrics(f*b/np.maximum(pred['disparity'],1e-3),gt_depth))
            elif a.dataset=='eigen':
                f,b=calibration(row['calibration'])
                metrics.update(depth_metrics(f*b/np.maximum(pred['disparity'],1e-3),eigen_gt(row)))
        results.append(metrics)
        if a.detailed_records or a.save_predictions:
            manifest.append({k:[public_path(vv) for vv in v] if k=='inputs' else public_path(v) if isinstance(v,Path) else v for k,v in row.items()})
        if a.save_predictions:
            directory=out/'predictions'/f'{i:06d}'; directory.mkdir(parents=True)
            if a.task!='depth': np.save(directory/'flow.npy',pred['flow'])
            if a.task!='flow': np.save(directory/'disparity.npy',pred['disparity'])
    if a.detailed_records or a.save_predictions:
        import csv
        keys=list(dict.fromkeys(k for r in results for k in r))
        with (out/'per_sample.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(results)
        (out/'sample_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    keys=list(dict.fromkeys(k for r in results for k in r))
    averages={k:float(np.mean([r[k] for r in results if r.get(k) is not None])) for k in keys if k!='sample_id' and not k.startswith('valid_')}
    write_metadata(out/'summary.json',averages)


if __name__=='__main__': main()
