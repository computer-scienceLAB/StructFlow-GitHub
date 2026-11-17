"""Original StructFlow flow and scalar visualizations."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb


def flow_rgb(flow, max_motion=256.0):
    """Render flow with the original UnRigidFlow HSV convention."""
    if max_motion is None:
        max_motion = float(np.max(flow))
    max_motion = max(float(max_motion), 1.0)
    u, v = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(np.square(u) + np.square(v))
    hue = np.mod(np.arctan2(v, u) / (2 * np.pi) + 1, 1)
    saturation = np.clip(magnitude * 8 / max_motion, 0, 1)
    value = np.clip(8 - saturation, 0, 1)
    return (hsv_to_rgb(np.stack([hue, saturation, value], axis=2)) * 255).astype(np.uint8)


def scalar_rgb(value, maximum=80.0, valid=None):
    if maximum <= 0:
        raise ValueError('maximum must be positive')
    rgb = (colormaps['magma'](np.clip(value/maximum,0,1))[...,:3]*255).astype(np.uint8)
    if valid is not None:
        rgb[~valid] = 0
    return rgb


def labeled_panel(images, labels):
    h,w = images[0].shape[:2]
    canvas = Image.new('RGB', (w*len(images), h+48), '#101820')
    draw = ImageDraw.Draw(canvas)
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError:
        font=ImageFont.load_default(size=18)
    for i,(image,label) in enumerate(zip(images,labels)):
        canvas.paste(Image.fromarray(image), (i*w,48))
        draw.text((i*w+12,14), label, fill='white', font=font)
    return canvas


def save_legend(path, task, maximum):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if task == 'flow':
        y,x=np.mgrid[-1:1:201j,-1:1:201j]
        rgb=flow_rgb(np.stack([x,y],-1).astype(np.float32)*maximum,maximum)
        rgb[np.hypot(x,y)>1]=255
        fig,ax=plt.subplots(figsize=(3,3)); ax.imshow(rgb); ax.axis('off')
        ax.set_title(f'Flow direction; saturation scale {maximum:g} px',fontsize=9)
    else:
        fig,ax=plt.subplots(figsize=(5,1.2)); fig.subplots_adjust(bottom=.5)
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,maximum),cmap='magma'),cax=ax,orientation='horizontal',label=task+'; values above scale saturated')
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
