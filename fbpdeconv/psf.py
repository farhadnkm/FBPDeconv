import cupy as cp
from cupyx.scipy.special import j0, j1
from cupyx.scipy.ndimage import shift
#import scipy.special
#import scipy.ndimage
#from scipy.interpolate import interp1d

from utils.cmath_cp import complex_sqrt, complex_sqrt_zero, cabs



class PSF3D_FastGibsonLanni:
    def __init__(self, config, Nx, Ny, Nz, channel_index=0, scale_lateral=1, scale_axial=1, **args_override) -> None:
        args_override = dict((k.lower(), v) for k, v in args_override.items())
        # Precision control
        self.num_basis          = config["PSF3D_FastGibsonLanni"]["num_basis"]  # Number of rescaled Bessels that approximate the phase function
        self.num_samples        = config["PSF3D_FastGibsonLanni"]["num_samples"] # Number of pupil samples along radial direction
        self.upsampling         = config["PSF3D_FastGibsonLanni"]["upsampling"]
        self.refocus_scan_range = args_override.get('refocus_scan_range', config["PSF3D_FastGibsonLanni"]["refocus_scan_range"])
        self.refocus_scan_step  = config["PSF3D_FastGibsonLanni"]["refocus_scan_step"]
        self.lateral_shift      = args_override.get('lateral_shift', config["PSF3D_FastGibsonLanni"]["lateral_shift"])
        # Microscope parameters
        self.NA                 = config["measurement"]["NA"]
        self.wavelength         = args_override.get("WL", config["measurement"]["WL"])
        self.M                  = config["measurement"]["MAGN"]        # magnification
        self.ns                 = config["measurement"]["Ns"]          # specimen refractive index (RI)
        self.ng0                = config["measurement"]["Ng0"]         # coverslip RI design value
        self.ng                 = config["measurement"]["Ng"]          # coverslip RI experimental value
        self.ni0                = config["measurement"]["Ni0"]         # immersion medium RI design value
        self.ni                 = config["measurement"]["Ni"]          # immersion medium RI experimental value
        self.ti0                = config["measurement"]["Ti0"]         # working distance (immersion medium thickness) design value
        self.tg0                = config["measurement"]["Tg0"]         # coverslip thickness design value
        self.tg                 = config["measurement"]["Tg"]          # coverslip thickness experimental value
        
        self.scale_lateral = scale_lateral
        self.scale_axial = scale_axial
        
        self.dr                 = args_override.get("dr", (config["measurement"]["DX"] / scale_lateral) / self.M)
        self.dz                 = args_override.get("DZ", config["measurement"]["DZ"] / scale_axial)
        self.pZ                 = config["PSF3D_FastGibsonLanni"]["pZ"]    # particle distance from coverslip
        
        
        self.dtype_f = args_override.get("dtype_f", cp.float32)
        self.dtype_c = cp.complex64 if self.dtype_f == cp.float32 else cp.complex128
        self.verbose     = args_override.get("verbose", 0)

        self.NX = Nx*scale_lateral
        self.NY = Ny*scale_lateral
        self.NZ = Nz*scale_axial
        self.z  = cp.arange(0, self.NZ)*self.dz  # microns, stage displacement away from best focus
        self.z -=  cp.mean(self.z)


        # NOTE: for backward compatibility, self.wavelength should be kept and WL should stay as a parameter in the config file
        if config["measurement"].keys().__contains__("Channels") and not args_override.keys().__contains__("WL"):
            wlem = config["measurement"]["Channels"][channel_index]["WLem"]
            self.wavelength = args_override.get("WL", wlem)
            if self.verbose > 0:
                print(f"Simulating the PSF for channel {channel_index}: {config["measurement"]["Channels"][channel_index]["name"]} - {config["measurement"]["Channels"][channel_index]["WLem"]*1000}nm emission wavelength")


        # Scaling factors for the Fourier-Bessel series expansion
        min_wavelength = 436e-3 # microns (empirically derived)
        self.scaling_factor = self.NA * (3 * cp.arange(1, self.num_basis + 1) - 2) * min_wavelength / self.wavelength

        # Place the origin at the center of the final PSF array
        self.x0 = (self.NX - 1) / 2
        self.y0 = (self.NY - 1) / 2

        # Find the maximum possible radius coordinate of the PSF array by finding the distance
        # from the center of the array to a corner
        self.max_radius = cp.round(cp.sqrt((self.NX - self.x0)**2 + (self.NY - self.y0)**2)) + 1

        # Radial coordinates, image space
        self.r = self.dr * cp.linspace(0, self.max_radius, int(self.max_radius * self.upsampling), dtype=self.dtype_f)

        # Radial coordinates, pupil space
        self.a = min([self.NA, self.ns, self.ni, self.ni0, self.ng, self.ng0]) / self.NA
        self.rho = cp.linspace(0, self.a, int(self.num_samples), dtype=self.dtype_f)
        self.NA2rho2 = self.NA * self.NA * self.rho * self.rho

        self.z_focus = 0.0
        self.z_focus_shift = 0.0
        self.setup_psf_shift_function()
        self.z_focus = self.find_psf_shift_by_function(0)
        #if self.pZ != 0:
            #self.z_focus = self.find_psf_shift(0.0, self.refocus_scan_range[0], self.refocus_scan_range[1], self.refocus_scan_step)
        
    def setup_psf_shift_function(self):
        y0 = self.find_psf_shift(0.0, -20, 20, 0.01)
        y1 = self.find_psf_shift(1000, -1400, -600, 0.01)
        self.tan_ = (y1 - y0) / 1000
        self.ofs_ = y0

    def find_psf_shift_by_function(self, z_shift):
        return self.tan_ * z_shift + self.ofs_


    def generate_psf_1D(self, z, z_shift=0.0):
        # Define the wavefront aberration
        OPDs = (self.pZ + z_shift) * complex_sqrt(self.ns * self.ns - self.NA2rho2) # OPD in the sample
        OPDi = (z.reshape(-1,1) + self.ti0) * complex_sqrt(self.ni * self.ni - self.NA2rho2) - self.ti0 * complex_sqrt(self.ni0 * self.ni0 - self.NA2rho2) # OPD in the immersion medium
        OPDg = self.tg * complex_sqrt(self.ng * self.ng - self.NA2rho2) - self.tg0 * complex_sqrt(self.ng0 * self.ng0 - self.NA2rho2) # OPD in the coverslip
        W    = 2 * cp.pi / self.wavelength * (OPDs + OPDi + OPDg)

        # Sample the phase
        # Shape is (number of z samples by number of rho samples)

        phase = cp.exp(1j * W)
        #plt.plot(np.abs(phase))

        # Define the basis of Bessel functions
        # Shape is (number of basis functions by number of rho samples)

        J = j0(self.scaling_factor.reshape(-1, 1) * self.rho)

        # Compute the approximation to the sampled pupil phase by finding the least squares
        # solution to the complex coefficients of the Fourier-Bessel expansion.
        # Shape of C is (number of basis functions by number of z samples).
        # Note the matrix transposes to get the dimensions correct.
        C, residuals, _, _ = cp.linalg.lstsq(J.T, phase.T, rcond=None)


        b = 2 * cp.pi * self.r.reshape(-1, 1) * self.NA / self.wavelength

        # Convenience functions for J0 and J1 Bessel functions
        J0 = lambda x: j0(x)
        J1 = lambda x: j1(x)

        # See equation 5 in Li, Xue, and Blu
        denom = self.scaling_factor * self.scaling_factor - b * b
        R = (self.scaling_factor * J1(self.scaling_factor * self.a) * J0(b * self.a) * self.a - b * J0(self.scaling_factor * self.a) * J1(b * self.a) * self.a)
        R /= denom


        # The transpose places the axial direction along the first dimension of the array, i.e. rows
        # This is only for convenience.
        PSF_rz = (R.dot(C)).T
        return PSF_rz

    def transform_psf_1D_to_3D_real(self, PSF_rz, dr):
        xy      = cp.mgrid[0:self.NY, 0:self.NX]
        r_pixel = cp.sqrt((xy[1] - self.x0) * (xy[1] - self.x0) + (xy[0] - self.y0) * (xy[0] - self.y0)) * dr

        PSF     = cp.zeros((self.NY, self.NX, self.NZ))
        for z_index in range(self.NZ):
            # Interpolate the radial PSF function
            # Evaluate the PSF at each value of r_pixel
            PSF[:,:, z_index] = cp.interp(r_pixel.ravel(), self.r, cp.abs(PSF_rz)[z_index, :]).reshape(self.NY, self.NX)
        return PSF

    def transform_psf_1D_to_3D_complex(self, PSF_rz, dr):
        xy      = cp.mgrid[0:self.NY, 0:self.NX]
        r_pixel = cp.sqrt((xy[1] - self.x0) * (xy[1] - self.x0) + (xy[0] - self.y0) * (xy[0] - self.y0)) * dr

        PSF     = cp.zeros((self.NY, self.NX, self.NZ), dtype=self.dtype_c)
        for z_index in range(self.NZ):
            # PSF[:,:, z_index] = cp.interp(r_pixel.ravel(), self.r, cp.real(PSF_rz)[z_index, :]).reshape(self.NY, self.NX) + 1.0j * cp.interp(r_pixel.ravel(), self.r, cp.imag(PSF_rz)[z_index, :]).reshape(self.NY, self.NX)
            PSF[:,:, z_index] = cp.interp(r_pixel.ravel(), self.r, cp.real(PSF_rz)[z_index, :]).reshape(self.NY, self.NX) + 1.0j * cp.interp(r_pixel.ravel(), self.r, cp.imag(PSF_rz)[z_index, :]).reshape(self.NY, self.NX)
        return PSF

    def find_psf_shift(self, z_shift, z_min=-50, z_max=0, z_step=0.1):
        z_test = cp.arange(z_min, z_max, z_step) + z_step
        psf = self.generate_psf_1D(z_test, z_shift)
        z_idx = cp.argmax(cabs(psf)[:, 0])
        return z_test[z_idx]
    
    def generate_psf_3D(self, return_complex=False, **kwargs):
        z = self.z.copy()
        dr = self.dr
        z_shift = 0.0
        scan_range = self.refocus_scan_range
        scan_step = self.refocus_scan_step
        lateral_shift = self.lateral_shift

        refocus = False
        for key in kwargs.keys():
            if key == 'z':
                z = kwargs[key].copy()
            elif key == 'dz':
                dz = kwargs[key]
                z  = cp.arange(0, self.NZ)*dz
                z -=  cp.mean(z)
            elif key == 'dr':
                dr = kwargs[key]
                self.r = dr * cp.linspace(0, self.max_radius, int(self.max_radius * self.upsampling))
            elif key == 'z_shift':
                z_shift = kwargs[key]
                refocus = kwargs.get('refocus', True)
            elif key == 'scan_range':
                scan_range = kwargs[key]
            elif key == 'scan_step':
                scan_step = kwargs[key]
            elif key == 'lateral_shift':
                lateral_shift = kwargs[key]
        
        if refocus:
            self.z_focus_shift = self.find_psf_shift_by_function(z_shift)#kwargs.get("psf_shift")#-z_shift#self.find_psf_shift(z_shift, scan_range[0], scan_range[1], scan_step)
        
        z += self.z_focus + self.z_focus_shift
        PSF_rz = self.generate_psf_1D(z, z_shift)

        if return_complex:
            PSF = self.transform_psf_1D_to_3D_complex(PSF_rz, dr)
        else:
            PSF = self.transform_psf_1D_to_3D_real(PSF_rz, dr)

        if lateral_shift != [0.0, 0.0]:
            PSF = shift(PSF, (lateral_shift[0], lateral_shift[1], 0))
        return PSF