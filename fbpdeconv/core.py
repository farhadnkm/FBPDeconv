import cupy as cp
from tqdm import tqdm


ZEROC64 = cp.array([0.0], dtype=cp.complex64)


def convn_a_bf_cp(a, bf):
    return cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftn(a) * bf))

def filtered_wiener_cp32(otf, alpha, n=0.0):
    return cp.where(cp.abs(otf) > alpha, cp.conj(otf)/ (otf*cp.conj(otf) + n), ZEROC64)

import cupyx.scipy.ndimage
def resample(dataset, scale=(1, 2, 2), order=0):
    """
    order:
    0 -> nearest
    1 -> bilinear
    2 -> cubic
    ...
    <= 5
    """
    return cupyx.scipy.ndimage.zoom(dataset, zoom=scale, order=order)

def block_reduce_2by2(input_):
    return 0.25*(input_[0::2, 0::2, :] + input_[1::2, 0::2, :] + input_[0::2, 1::2, :] + input_[1::2, 1::2, :])


def RL_cp32_resample(image_stack, psf, psf_up, iters=50):
    fourier_phase_correction = [0,0]#[-(s%2) if s!=1 else 0 for s in psf.shape]
    bp = cp.fft.fftn(cp.roll(psf, fourier_phase_correction))
    otf = cp.fft.fftn(psf_up)
    out = resample(image_stack, scale=(2, 2, 1), order=1)
    for _ in tqdm(range(iters)):
        out_otf = block_reduce_2by2(convn_a_bf_cp(out, otf))
        out *= resample(convn_a_bf_cp(image_stack / out_otf, bp), (2, 2, 1), 1)
    return out


def RL_TW_cp32_resample(image_stack, psf, psf_upsampled, iters=5, otf_thres=1e-2, wiener_param=2e-3, alpha=0.1, gamma=1.0, upsample=1):
    psf /= cp.sum(psf)
    psf_upsampled /= cp.sum(psf_upsampled)
    otf = cp.fft.fftn(psf_upsampled)
    fourier_phase_correction = [-((s+1)%2) if s!=1 else 0 for s in psf.shape]
    print(fourier_phase_correction)
    out = resample(image_stack, scale=(upsample, upsample, 1), order=1)
    for i in tqdm(range(iters)):
        c = (0.1/(alpha)) * (i + 0.1*gamma*alpha) / (i + gamma*alpha) if gamma > 1e-8 else 1
        bp = filtered_wiener_cp32(cp.fft.fftn(cp.roll(psf, fourier_phase_correction)), otf_thres, wiener_param*c)
        out *= resample(convn_a_bf_cp(image_stack/block_reduce_2by2(convn_a_bf_cp(out, otf)), bp), (upsample, upsample, 1), 1)
        out[out<0] = 0
        out += alpha
    return out