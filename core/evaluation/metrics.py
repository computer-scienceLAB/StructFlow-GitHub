import numpy as np


def mean_on(x,mask):
    return float(x[mask].mean()) if mask.any() else None


def flow_metrics(pred,gt,valid,noc=None):
    if pred.shape!=gt.shape or not np.isfinite(pred).all(): raise ValueError('Invalid flow prediction')
    e=np.linalg.norm(pred-gt,axis=-1); mag=np.linalg.norm(gt,axis=-1)
    out=dict(epe_all=mean_on(e,valid), f1_5pct=mean_on(((e>3)&(e>0.05*mag)).astype(float)*100,valid),
             f1_9pct=mean_on(((e>3)&(e>0.09*mag)).astype(float)*100,valid),valid_flow_pixels=int(valid.sum()))
    if noc is not None:
        match=valid&noc; unmatched=valid&~noc
        out.update(epe_match=mean_on(e,match),epe_unmatch=float(e[unmatched].mean()) if unmatched.any() else 0.0,
                   unmatched_valid_percent=float(unmatched.sum()/valid.sum()*100))
    return out


def disparity_metrics(pred,gt,valid):
    if pred.shape!=gt.shape or not np.isfinite(pred).all(): raise ValueError('Invalid disparity prediction')
    e=np.abs(pred-gt)
    return dict(disp_mae=mean_on(e,valid),disp_d1_bad3=mean_on(((e>3)&(e>0.05*np.abs(gt))).astype(float)*100,valid),valid_disparity_pixels=int(valid.sum()))


def depth_metrics(pred,gt):
    valid=(gt>0)&(gt<80)&np.isfinite(gt)
    h,w=gt.shape; crop=np.zeros_like(valid)
    crop[int(.40810811*h):int(.99189189*h),int(.03594771*w):int(.96405229*w)]=True
    valid &= crop
    if not valid.any(): raise ValueError('No valid cropped depth ground truth')
    if not np.isfinite(pred).all(): raise ValueError('Non-finite depth prediction')
    a=gt[valid]; b=pred[valid].clip(1e-3,80)
    threshold=np.maximum(a/b,b/a)
    return dict(abs_rel=float(np.mean(np.abs(a-b)/a)),sq_rel=float(np.mean((a-b)**2/a)),
                rmse=float(np.sqrt(np.mean((a-b)**2))),rmse_log=float(np.sqrt(np.mean((np.log(a)-np.log(b))**2))),
                delta1=float(np.mean(threshold<1.25)),delta2=float(np.mean(threshold<1.25**2)),delta3=float(np.mean(threshold<1.25**3)),valid_depth_pixels=int(valid.sum()))
