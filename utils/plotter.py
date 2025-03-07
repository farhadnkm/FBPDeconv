import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def show_image_3D(img, size_scale=(5, 5), dr=(1, 1, 1), gaps=(0.1, 0.1), x_idx=None, y_idx=None, z_idx=None, vmin=None, vmax=None, projection=('plane', 'plane', 'plane'), interpolation='none', cmap='gray'):
    """
    3D stack viewer,
    img: image stack, the shape should be YXZ or YXZC for RGB stacks.
    """
    if x_idx is None: x_idx = img.shape[1]//2
    if y_idx is None: y_idx = img.shape[0]//2
    if z_idx is None: z_idx = img.shape[2]//2
    
    dy, dx, dz = dr[0]*img.shape[0], dr[1]*img.shape[1], dr[2]*img.shape[2]

    ratios = [(dy, dz), (dx, dz)]
    size = (((dx + dz)/np.sqrt(dy**2 + dz**2))*size_scale[0], ((dy + dz)/np.sqrt(dx**2 + dz**2))*size_scale[1])

    fig = plt.figure(figsize=size)
    gs = GridSpec(2, 2, width_ratios=ratios[0], height_ratios=ratios[1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])

    def makez(projection):
        if projection == 'plane':
            return img[:, :, z_idx]
        elif projection == 'mean':
            return np.mean(img, axis=2)
        elif projection == 'max':
            return np.max(img, axis=2)
        elif projection == 'min':
            return np.min(img, axis=2)
    
    def makex(projection):
        if projection == 'plane':
            return img[:, x_idx, ::-1]
        elif projection == 'mean':
            return np.mean(img[:, :, ::-1], axis=1)
        elif projection == 'max':
            return np.max(img[:, :, ::-1], axis=1)
        elif projection == 'min':
            return np.min(img[:, :, ::-1], axis=1)
    
    def makey(projection):
        if projection == 'plane':
            return np.moveaxis(img[y_idx, :, ::-1], 0, 1)
        elif projection == 'mean':
            return np.moveaxis(np.mean(img[:, :, ::-1], axis=0), 0, 1)
        elif projection == 'max':
            return np.moveaxis(np.max(img[:, :, ::-1], axis=0), 0, 1)
        elif projection == 'min':
            return np.moveaxis(np.min(img[:, :, ::-1], axis=0), 0, 1)
        
    ax1.imshow(makez(projection[2]), cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation=interpolation)
    ax2.imshow(makex(projection[1]), cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation=interpolation)
    ax3.imshow(makey(projection[0]), cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation=interpolation)

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    gs.tight_layout(fig, h_pad=gaps[0], w_pad=gaps[1])
    plt.show()
