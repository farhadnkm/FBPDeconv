# from https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
"""
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
"""


# Note: the script is modified for alpha=1 and the parameter is removed.

import numpy as np
from utils.kernels import P5, P7, P27
from scipy import fftpack
from scipy.ndimage import convolve

def wrap_boundary_liu_2D(img, img_size):

    (H,W) = np.shape(img)
    H_w = img_size[0] - H
    W_w = img_size[1] - W

    HG = img[:,:]

    r_A = np.zeros((2 + H_w,W))
    r_A[:1, :] = HG[-1:,:]
    r_A[-1:,:] = HG[:1, :]
    r_A[1:-1, 0] = np.linspace(r_A[0, 0], r_A[-1, 0], int(H_w))
    r_A[1:-1,-1] = np.linspace(r_A[0,-1], r_A[-1,-1], int(H_w))

    r_B = np.zeros((H, 2+W_w))
    r_B[:, :1] = HG[:, -1:]
    r_B[:,-1:] = HG[:, :1]
    r_B[0, 1:-1] = np.linspace(r_B[0, 0], r_B[0, -1], int(W_w))
    r_B[-1,1:-1] = np.linspace(r_B[-1,0], r_B[-1,-1], int(W_w))
    
    A = min_laplace_2D(r_A)
    B = min_laplace_2D(r_B)

    r_C = np.zeros((2 + H_w, 2 + W_w))
    r_C[:1, :] = B[-1:, :]
    r_C[-1:, :] = B[:1, :]
    r_C[:, :1] = A[:, -1:]
    r_C[:, -1:] = A[:, :1]

    C = min_laplace_2D(r_C)

    #return C
    A = A[0:-2, :]
    B = B[:, 1:-1]
    C = C[1:-1, 1:-1]
    
    ret = np.vstack((np.hstack((img, B)), np.hstack((A, C))))
    return ret

def wrap_boundary_liu_3D(vol, out_size):
    H, W, D = np.shape(vol)
    H_w = out_size[0] - H
    W_w = out_size[1] - W
    D_w = out_size[2] - D

    HG = vol.copy()

    r_A = np.zeros((H_w+2, W, D))
    r_A[0, :, :] = HG[-1,:, :]
    r_A[-1,:, :] = HG[0, :, :]
    r_A[1:-1, 0, 0] = np.linspace(r_A[0, 0, 0], r_A[-1, 0, 0], int(H_w))
    r_A[1:-1, 0,-1] = np.linspace(r_A[0, 0,-1], r_A[-1, 0,-1], int(H_w))
    r_A[1:-1,-1, 0] = np.linspace(r_A[0,-1, 0], r_A[-1,-1, 0], int(H_w))
    r_A[1:-1,-1,-1] = np.linspace(r_A[0,-1,-1], r_A[-1,-1,-1], int(H_w))
    r_A[:, 0, :] = min_laplace_2D(r_A[:, 0, :])
    r_A[:,-1, :] = min_laplace_2D(r_A[:,-1, :])
    r_A[:, :, 0] = min_laplace_2D(r_A[:, :, 0])
    r_A[:, :,-1] = min_laplace_2D(r_A[:, :,-1])

    A = min_laplace_3D(r_A)

    r_B = np.zeros((H, W_w+2, D))
    r_B[:, 0, :] = HG[:,-1, :]
    r_B[:,-1, :] = HG[:, 0, :]
    r_B[0, 1:-1, 0] = np.linspace(r_B[0, 0, 0], r_B[0, -1, 0], int(W_w))
    r_B[0, 1:-1,-1] = np.linspace(r_B[0, 0,-1], r_B[0, -1,-1], int(W_w))
    r_B[-1,1:-1, 0] = np.linspace(r_B[-1,0, 0], r_B[-1,-1, 0], int(W_w))
    r_B[-1,1:-1,-1] = np.linspace(r_B[-1,0,-1], r_B[-1,-1,-1], int(W_w))
    r_B[0, :, :] = min_laplace_2D(r_B[0, :, :])
    r_B[-1,:, :] = min_laplace_2D(r_B[-1,:, :])
    r_B[:, :, 0] = min_laplace_2D(r_B[:, :, 0])
    r_B[:, :,-1] = min_laplace_2D(r_B[:, :,-1])
    
    B = min_laplace_3D(r_B)

    r_C = np.zeros((H, W, D_w+2))
    r_C[:, :, 0] = HG[:, :,-1]
    r_C[:, :,-1] = HG[:, :, 0]
    r_C[0,  0,1:-1] = np.linspace(r_C[0,  0, 0], r_C[0,  0,-1], int(D_w))
    r_C[0, -1,1:-1] = np.linspace(r_C[0, -1, 0], r_C[0, -1,-1], int(D_w))
    r_C[-1, 0,1:-1] = np.linspace(r_C[-1, 0, 0], r_C[-1, 0,-1], int(D_w))
    r_C[-1,-1,1:-1] = np.linspace(r_C[-1,-1, 0], r_C[-1,-1,-1], int(D_w))
    r_C[0, :, :] = min_laplace_2D(r_C[0, :, :])
    r_C[-1,:, :] = min_laplace_2D(r_C[-1,:, :])
    r_C[:, 0, :] = min_laplace_2D(r_C[:, 0, :])
    r_C[:,-1, :] = min_laplace_2D(r_C[:,-1, :])

    C = min_laplace_3D(r_C)

    r_AB = np.zeros((H_w+2, W_w+2, D))
    r_AB[:, 0, :] = A[:,-1, :]
    r_AB[:,-1, :] = A[:, 0, :]
    r_AB[0, :, :] = B[-1,:, :]
    r_AB[-1,:, :] = B[0, :, :]
    r_AB[:, :, 0] = min_laplace_2D(r_AB[:, :, 0])
    r_AB[:, :,-1] = min_laplace_2D(r_AB[:, :,-1])
    
    AB = min_laplace_3D(r_AB)

    r_AC = np.zeros((H_w+2, W, D_w+2))
    r_AC[:, :, 0] = A[:, :,-1]
    r_AC[:, :,-1] = A[:, :, 0]
    r_AC[0, :, :] = C[-1,:, :]
    r_AC[-1,:, :] = C[0, :, :]
    r_AC[:, 0, :] = min_laplace_2D(r_AC[:, 0, :])
    r_AC[:,-1, :] = min_laplace_2D(r_AC[:,-1, :])
    
    AC = min_laplace_3D(r_AC)

    r_BC = np.zeros((H, W_w+2, D_w+2))
    r_BC[:, :, 0] = B[:, :,-1]
    r_BC[:, :,-1] = B[:, :, 0]
    r_BC[:, 0, :] = C[:,-1, :]
    r_BC[:,-1, :] = C[:, 0, :]
    r_BC[0, :, :] = min_laplace_2D(r_BC[0, :, :])
    r_BC[-1,:, :] = min_laplace_2D(r_BC[-1,:, :])
    
    BC = min_laplace_3D(r_BC)

    r_ABC = np.zeros((H_w+2, W_w+2, D_w+2))
    r_ABC[:, :, 0] = AB[:, :,-1]
    r_ABC[:, :,-1] = AB[:, :, 0]
    r_ABC[:, 0, :] = AC[:,-1, :]
    r_ABC[:,-1, :] = AC[:, 0, :]
    r_ABC[0, :, :] = BC[-1,:, :]
    r_ABC[-1,:, :] = BC[0, :, :]
    
    ABC = min_laplace_3D(r_ABC)
    
    A = A[1:-1, ...]
    B = B[:, 1:-1, :]
    C = C[..., 1:-1]
    AB = AB[1:-1, 1:-1, :]
    AC = AC[1:-1, :, 1:-1]
    BC = BC[:, 1:-1, 1:-1]
    ABC = ABC[1:-1, 1:-1, 1:-1]

    out = np.concatenate([np.concatenate([np.concatenate([vol, A], axis=0), np.concatenate([B, AB], axis=0)], axis=1),
                          np.concatenate([np.concatenate([C,  AC], axis=0), np.concatenate([BC,ABC],axis=0)], axis=1)], axis=2)
    return out

def min_laplace_2D(boundary):
    H, W = boundary.shape
    lap = -convolve(boundary, P5, cval=0, mode='constant')[1:-1, 1:-1]
    l_dst = fftpack.dst(lap, type=1)/2
    l_dst = (fftpack.dst(l_dst.T,type=1)/2).T
    
    # compute Eigen Values
    x, y = np.meshgrid(np.linspace(1/(W-1),1, W-2), np.linspace(1/(H-1),1, H-2))
    denom = 2*(np.cos(np.pi*x) + np.cos(np.pi*y)) - 4
    l_dst = l_dst/denom
    
    l_dst = fftpack.idst(l_dst*2,type=1)
    img_tt = (fftpack.idst(l_dst.T*2,type=1)).T / (4*(H-1)*(W-1))
    out = boundary.copy()
    out[1:-1, 1:-1] = img_tt
    return out

def min_laplace_3D(boundary):
    lap = -convolve(boundary, P7, cval=0, mode='constant')[1:-1, 1:-1, 1:-1]
    
    first = (2, 1, 0)
    sec = (2, 0, 1)
    last = (1, 2, 0)

    l_dst = fftpack.dst(lap, type=1) /2
    l_dst = np.transpose(fftpack.dst(np.transpose(l_dst, first), type=1), first) /2
    l_dst = np.transpose(fftpack.dst(np.transpose(l_dst, sec),type=1), sec) /2
    
    H, W, D = l_dst.shape
    # compute Eigen Values
    x, y, z = np.meshgrid(np.linspace(1/(W),1-1/(W), W), np.linspace(1/(H),1-1/(H), H), np.linspace(1/(D),1-1/(D), D))
    denom = 2*(np.cos(np.pi*x) + np.cos(np.pi*y) + np.cos(np.pi*z)) - 6
    l_dst = l_dst/denom
    
    l_dst = fftpack.idst(l_dst*2,type=1)
    l_dst = np.transpose(fftpack.idst(np.transpose(l_dst*2, first),type=1), first)
    img_tt = np.transpose(fftpack.idst(np.transpose(l_dst*2, sec),type=1), sec) / (8*(H+1)*(W+1)*(D+1))
    out = boundary.copy()
    out[1:-1, 1:-1, 1:-1] = np.transpose(img_tt, last)
    return out