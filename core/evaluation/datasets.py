"""KITTI 15, KITTI 12, FlyingThings3D, Spring, Sintel and Eign Split data loaders."""
from pathlib import Path
import numpy as np
from PIL import Image


def samples(dataset, root, split=None, flow_root=None):
    root=Path(root); rows=[]
    if dataset in ('kitti2015','kitti2012'):
        left,right=('image_2','image_3') if dataset=='kitti2015' or (root/'image_2').is_dir() else ('colored_0','colored_1')
        for gt in sorted((root/'flow_occ').glob('*_10.png')):
            sid=gt.stem[:-3]
            row=dict(id=sid, inputs=[root/left/f'{sid}_10.png',root/left/f'{sid}_11.png',root/right/f'{sid}_10.png',root/right/f'{sid}_11.png'],
                     flow=gt, noc=root/'flow_noc'/gt.name)
            if dataset=='kitti2015':
                row.update(disparity=root/'disp_occ_0'/gt.name,calibration=root/'calib_cam_to_cam'/f'{sid}.txt')
            rows.append(row)
        expected=200 if dataset=='kitti2015' else 194
        if len(rows)!=expected:
            raise ValueError(f'{dataset}: expected {expected} annotated scenes, found {len(rows)}')
    elif dataset=='flyingthings3d':
        if split is None:
            raise ValueError('FlyingThings3D requires the exact split file; no implicit subset selection.')
        keys=[s.strip() for s in Path(split).read_text().splitlines() if s.strip() and not s.startswith('#')]
        if len(keys)!=len(set(keys)):
            raise ValueError('Duplicate split entries')
        for key in sorted(keys):
            pack,scene,frame=key.split('|'); nxt=f'{int(frame)+1:04d}'
            rgb=root/f'frames_{pack}'/scene
            rows.append(dict(id=key,inputs=[rgb/'left'/f'{frame}.png',rgb/'left'/f'{nxt}.png',rgb/'right'/f'{frame}.png',rgb/'right'/f'{nxt}.png'],
                             flow=root/'optical_flow'/scene/'into_future'/'left'/f'OpticalFlowIntoFuture_{frame}_L.pfm',
                             disparity=root/'disparity'/scene/'left'/f'{frame}.pfm'))
    elif dataset=='spring':
        for scene in sorted(root.iterdir()):
            if not scene.is_dir(): continue
            for im in sorted((scene/'frame_left').glob('*.png')):
                frame=im.stem.split('_')[-1]; nxt=f'{int(frame)+1:04d}'
                inputs=[im,scene/'frame_left'/f'frame_left_{nxt}.png',scene/'frame_right'/f'frame_right_{frame}.png',scene/'frame_right'/f'frame_right_{nxt}.png']
                if not inputs[1].exists(): continue
                rows.append(dict(id=f'{scene.name}_{frame}',inputs=inputs,flow=scene/'flow_FW_left'/f'flow_FW_left_{frame}.flo5',disparity=scene/'disp1_left'/f'disp1_left_{frame}.dsp5'))
    elif dataset=='sintel':
        if flow_root is None: raise ValueError('Sintel requires --flow-root for optical-flow annotations.')
        for scene in sorted((root/'final_left').iterdir()):
            if not scene.is_dir(): continue
            for im in sorted(scene.glob('*.png')):
                frame=int(im.stem.split('_')[-1]); nxt=f'frame_{frame+1:04d}.png'
                if not (scene/nxt).exists(): continue
                rows.append(dict(id=f'{scene.name}/{im.stem}',inputs=[im,scene/nxt,root/'final_right'/scene.name/im.name,root/'final_right'/scene.name/nxt],flow=Path(flow_root)/scene.name/f'{im.stem}.flo'))
    elif dataset=='eigen':
        if split is None: raise ValueError('Eigen evaluation requires the 697-image split file.')
        for line in Path(split).read_text().splitlines():
            if not line.strip(): continue
            sequence,frame,side=line.split()
            if side!='l': raise ValueError('This source protocol expects left-camera Eigen samples.')
            base=root/sequence; frame=f'{int(frame):010d}'
            inputs=[]; modes=[]
            for camera in ['image_02','image_03']:
                curr=base/camera/'data'/f'{frame}.png'
                nxt=base/camera/'data'/f'{int(frame)+1:010d}.png'
                prev=base/camera/'data'/f'{int(frame)-1:010d}.png'
                neighbor=nxt if nxt.exists() else prev if prev.exists() else curr
                inputs.extend([curr,neighbor]); modes.append('next' if neighbor==nxt else 'previous' if neighbor==prev else 'self')
            rows.append(dict(id=f'{sequence}/{frame}',inputs=inputs,calibration=base.parent/'calib_cam_to_cam.txt',
                             velo_calibration=base.parent/'calib_velo_to_cam.txt',velodyne=base/'velodyne_points'/'data'/f'{frame}.bin',neighbors=modes))
        if len(rows)!=697: raise ValueError(f'Expected 697 Eigen samples, found {len(rows)}')
    else:
        raise ValueError(dataset)
    if not rows: raise ValueError('No evaluation samples found')
    return rows


def check_files(rows):
    missing=[]
    for row in rows:
        for key,value in row.items():
            for p in (value if key=='inputs' else [value]):
                if isinstance(p,Path) and not p.is_file(): missing.append(str(p))
    if missing:
        raise FileNotFoundError(f'{len(missing)} required files missing; first: {missing[:10]}')


def calib_values(path):
    result={}
    for line in Path(path).read_text().splitlines():
        if ':' not in line: continue
        k,v=line.split(':',1)
        try: result[k]=np.array([float(x) for x in v.split()],np.float32)
        except ValueError: pass
    return result


def eigen_gt(row):
    # Preserve the archived Eigen evaluator's projection, floor indexing and nearest z-buffer.
    cam=calib_values(row['calibration']); velo=calib_values(row['velo_calibration'])
    p=cam.get('P_rect_02',cam.get('P2')).reshape(3,4)
    r=np.eye(4,dtype=np.float32); r[:3,:3]=cam['R_rect_00'].reshape(3,3)
    tr=np.eye(4,dtype=np.float32)
    if 'Tr_velo_to_cam' in velo: tr[:3,:4]=velo['Tr_velo_to_cam'].reshape(3,4)
    else: tr[:3,:3]=velo['R'].reshape(3,3); tr[:3,3]=velo['T'].reshape(3)
    points=np.fromfile(row['velodyne'],np.float32).reshape(-1,4)
    pts=np.concatenate([points[:,:3],np.ones((len(points),1),np.float32)],1)
    projected=(p@r@tr@pts.T).T
    projected=projected[projected[:,2]>0]; z=projected[:,2]
    x=projected[:,0]/z; y=projected[:,1]/z
    w,h=Image.open(row['inputs'][0]).size
    keep=(x>=0)&(x<w)&(y>=0)&(y<h)
    x=x[keep].astype(np.int32); y=y[keep].astype(np.int32); z=z[keep]
    depth=np.full((h,w),np.inf,np.float32)
    np.minimum.at(depth,(y,x),z)
    depth[~np.isfinite(depth)]=0
    return depth
