import numpy as np
import cupy as cp

import tifffile

from fbpdeconv.core import RL_TW_cp32_resample, RL_cp32_resample
from fbpdeconv.psf import PSF3D_FastGibsonLanni

import os

from utils.config import load_config
from utils.padding.padding_helper import pad, unpad
from utils.plotter import show_image_3D
from utils.image import normalize_perc



# import data
img_raw = np.flip(tifffile.imread(os.path.normpath(r"some address.tif")).astype(np.float64), axis=0)
img_raw = np.moveaxis(img_raw, 0, -1)
config = load_config(os.path.normpath(r"test/config.json"))
shape_raw = img_raw.shape

# pad images 
img = pad(img_raw, (10, 10, 10), verbose=False)
Ny, Nx, Nz = img.shape


# simulate PSF(s)
psf_generator = PSF3D_FastGibsonLanni(config=config, Nx=Nx, Ny=Ny, Nz=Nz, scale_lateral=1, channel_index=1, dtype_f=cp.float64)
psf1 = cp.abs(psf_generator.generate_psf_3D())**2
psf_generator = PSF3D_FastGibsonLanni(config=config, Nx=Nx, Ny=Ny, Nz=Nz, scale_lateral=2, channel_index=1, dtype_f=cp.float64)
psf2 = cp.abs(psf_generator.generate_psf_3D())**2


# reconstruct
# convensional RL algorithm
out = cp.real(RL_cp32_resample(cp.array(img, dtype=cp.complex64), psf1.astype(cp.complex64), psf2.astype(cp.complex64), iters=5000)).get()
out = unpad(out, shape_raw)
show_image_3D(normalize_perc(out, (0.01, 99.99)), (10, 10), dr=(1, 1, 1), gaps=(0.5, 0.5), 
              z_idx=None, x_idx=None, y_idx=None, projection=('max', 'max', 'max'), 
              interpolation='none', cmap='gray', vmin=0, vmax=1.)


# FIREBAL
out = cp.real(RL_TW_cp32_resample(cp.array(img, dtype=cp.complex64), psf1.astype(cp.complex64), psf2.astype(cp.complex64), 
                                  iters=20, otf_thres=1e-2, wiener_param=1e-3, alpha=0.1, gamma=100, upsample=2)).get()
out = unpad(out, shape_raw)
show_image_3D(out, (10, 10), dr=(1, 1, 1), gaps=(0.5, 0.5), 
              z_idx=None, x_idx=None, y_idx=None, projection=('max', 'max', 'max'), 
              interpolation='none', cmap='gray', vmin=0, vmax=1.)
