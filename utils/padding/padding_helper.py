import numpy as np
from time import time
from utils.padding.padding import wrap_boundary_liu_3D


def pad(image_stack, output_shape, verbose=True):
    if verbose: 
        print('padding...')
        t = time()
    Nx0, Ny0, Nz0 = image_stack.shape
    Nx,  Ny,  Nz  = output_shape

    image_stack_pad = wrap_boundary_liu_3D(image_stack, (Nx, Ny, Nz))
    image_stack_pad = np.roll(image_stack_pad, (Nx-Nx0)//2, 0)
    image_stack_pad = np.roll(image_stack_pad, (Ny-Ny0)//2, 1)
    image_stack_pad = np.roll(image_stack_pad, (Nz-Nz0)//2, 2)

    if verbose: print(f'Padding completed - shape_in:{(Nx0, Ny0, Nz0)}, shape_padded:{(Nx,  Ny,  Nz)} | elapsed time:', time() - t)
    return image_stack_pad

def unpad(stack, output_shape):
    slices = []
    in_shape = stack.shape
    for i in range(len(output_shape)):
        slices.append(slice((in_shape[i]-output_shape[i])//2,(in_shape[i]+output_shape[i])//2))
    return stack[*slices]

