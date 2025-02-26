import os
import numpy as np
from matplotlib import pyplot as plt
import rawpy as rawpy_lib
#from scipy.ndimage.filters import gaussian_filter as gaussMd
from scipy.ndimage import gaussian_filter as gaussMd

class Ispeximage(object):
    """
    An instance of Ispeximage class is created for each image file.
    The class contains methods to process the image file and save the processed data.
    """
    def __init__(self,
                 dng_path=None,
                 save_path=None,
                 camera='apple_iphone_mini_13',
                 output_plots=False):
        """
        Initialize the Ispeximage object with the path to the dng image file,
          camera profile, and output directory (if producing plots).

        param dng_path: str, path to the dng image file
        param save_path: str, path to the output directory
        param phone: str, name of camera profile
        param output_plots: bool, whether to produce plots
        """
        self.dng_path = dng_path            # source RAW image file
        self.label = os.path.basename(dng_path).split(".")[0]
        self.save_path = save_path          # output directory for plots, data files
        self.output_plots = output_plots    # whether to produce plots
        self.img_raw = None                 # raw image data
        self.img_post = None                # raw image data mapped to RGB, for visualisation purposes only
        self.camera = camera                # camera model, used to load calibration profile
        self.slice_Qm = None                # Qx pixel selection window
        self.slice_Qp = None                # 
        self.Qm_sliced = None               # Qx raw data block
        self.Qp_sliced = None               # 
        self.Qm_background_corrected = None # Qx raw data block, background corrected
        self.Qp_background_corrected = None # 
        self.background_level = None        # Background noise level in the image, in brightness units
        self.wavelengths_split_Qp = None    # Wavelength mapped to each pixel in the sliced Qx 4d array
        self.wavelengths_split_Qm = None    #
        self.Qp_RGBG = None                 # RGBG values for each pixel in the Qx image after previous processing steps
        self.Qm_RGBG = None                 #
        self.Qp_stacked_RGB_mean = None     # Qx RGB values averaged along the slit dimension
        self.Qm_stacked_RGB_mean = None     #
        try:
            self.wl_calib_qp = np.load(os.path.join("cameras", self.camera, "wavelength_calibration_Qp.npy"))
            self.wl_calib_qm = np.load(os.path.join("cameras", self.camera, "wavelength_calibration_Qm.npy"))
        except IOError:
            self.wl_calib_qp = None
            self.wl_calib_qm = None
            raise

        self.process()

    def process(self):
        """
        Process the image file in memory
        """
        # read the raw image
        with rawpy_lib.imread(self.dng_path) as img:
            self.img_raw = img.raw_image.astype(np.float64)   #  was 'data'
            bayer_map = img.raw_colors
            self.img_post = img.postprocess()

        # Slice the image to extract the Qp and Qm spectral blocks
        self.slice_Qp, self.slice_Qm = self.find_spectrum_slices(model="SPIE")  # FIXME: hardcoded slices -> find automatically
        self.Qp_sliced, self.Qm_sliced = self.img_raw[self.slice_Qp], self.img_raw[self.slice_Qm]

        # generate the bayer map (pixel to RGBG colour mapping)
        bayer_Qp, bayer_Qm = bayer_map[self.slice_Qp], bayer_map[self.slice_Qm]

        # FIXME: improve noise level estimate 
        #  - after field corrections, possibly 2d interpolate the sliced area with some buffer?
        #  - would this alleviate the need for flat field corrections?
        #  - it would be nice to find the projected spectrum by looking at image contrasts
        #  - Optimal Estimation method might also work here.
        self.background_level = np.average(np.mean(self.Qp_sliced[:, :250], axis=1))
        # subtract noise level from the image
        self.Qp_background_corrected = self.Qp_sliced - self.background_level
        self.Qm_background_corrected = self.Qm_sliced - self.background_level

        # Map pixels to wavelength grid
        x = np.arange(self.Qp_sliced.shape[1])   # along-spectrum axis length (shorter to longer wavelength), same for Qp and Qm
        xp = np.repeat(x[:,np.newaxis], bayer_Qp.shape[0], axis=1).T  # along-spectrum axis repeated for each line in Qp
        xm = np.repeat(x[:,np.newaxis], bayer_Qm.shape[0], axis=1).T  # along-spectrum axis repeated for each line in Qm
        yp = np.arange(self.Qp_background_corrected.shape[0])  # along-slit axis length Qp
        ym = np.arange(self.Qm_background_corrected.shape[0])  # along-slit axis length Qm
        coeff_fit = np.array([np.polyval(c, yp) for c in self.wl_calib_qp]).T
        wavelengths_Qp = np.array([np.polyval(c_fit, x) for c_fit in coeff_fit])
        coeff_fit = np.array([np.polyval(c, ym) for c in self.wl_calib_qm]).T
        wavelengths_Qm = np.array([np.polyval(c_fit, x) for c_fit in coeff_fit])

        # Demosaick using the bayer map (mapping pixels to RGBG colours)
        # - wavelengths_split_Qx are the wavelength mapping for each pixel in the Qx image, 
        #   correcting for the encoded smile effect.
        # - RGBG_Qx are the RGBG values for each pixel in the Qx image
        self.wavelengths_split_Qp, self.RGBG_Qp, xp_split = self.demosaick(bayer_Qp, [wavelengths_Qp, self.Qp_background_corrected, xp])
        self.wavelengths_split_Qm, self.RGBG_Qm, xm_split = self.demosaick(bayer_Qm, [wavelengths_Qm, self.Qm_background_corrected, xm])

        # smoothing using a gaussian kernel along the spectral dimension
        self.RGBG_Qp_smoothed = self._gauss_nan(self.RGBG_Qp, sigma=(0,0,3))  # sigma axes: RGBG channel, along-slit, along-spectrum
        self.RGBG_Qm_smoothed = self._gauss_nan(self.RGBG_Qm, sigma=(0,0,3))

        # Through interpolation, translate 1 pixel in RGBG space to 2 pixels in RGB space
        # # lambdarange = resultant wavelength grid
        # all_interpolated_Qp = interpolated spectra for each line in Qp/Qm image slice
        lambdarange, Qp_RGBG = self.interpolate_multi(self.wavelengths_split_Qp, self.RGBG_Qp_smoothed)
        lambdarange, Qm_RGBG = self.interpolate_multi(self.wavelengths_split_Qm, self.RGBG_Qm_smoothed)
        
        # stack the RGBG into a WL,R,G,B array (4xN) 
        self.Qp_stacked_RGB = self.stack(lambdarange, Qp_RGBG)  # RGB radiance in arbitrary units
        self.Qm_stacked_RGB = self.stack(lambdarange, Qm_RGBG)  # RGB radiance in arbitrary units

        if self.output_plots:
            self.plot_bounding_boxes()
            self.plot_spectra()

    def plot_spectra(self):
        """
        Plot radiance spectra in arbitrary units
        """
        # Qp stack
        plt.figure(figsize=(6,2))
        for j, c in enumerate("rgb", 1):
            plt.plot(self.Qp_stacked_RGB[0], self.Qp_stacked_RGB[j], c=c)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Radiance [a.u.]")
        plt.grid(ls="--")
        plt.ylim(-5, np.nanmax(self.Qp_stacked_RGB[1:])*1.05)
        plt.xlim(390, 700)

        if self.output_plots:
            plt.savefig(os.path.join(self.save_path, f"{self.label}_Qp.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Qm stack
        plt.figure(figsize=(6,2))
        for j, c in enumerate("rgb", 1):
            plt.plot(self.Qm_stacked_RGB[0], self.Qm_stacked_RGB[j], c=c)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Radiance [a.u.]")
        plt.grid(ls="--")
        plt.ylim(-5, np.nanmax(self.Qm_stacked_RGB[1:])*1.05)
        plt.xlim(390, 700)
        if self.output_plots:
            plt.savefig(os.path.join(self.save_path, f"{self.label}_Qm.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Spectrum plot
        plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})
        plt.figure(figsize=(10, 4))  # Wider figure

        # Use a loop to plot each spectrum with a thicker line for visibility
        for j, color in zip(range(1, 4), ['red', 'green', 'blue']):  # Explicit color names for clarity
            plt.plot(self.Qp_stacked_RGB[0], self.Qp_stacked_RGB[j]+self.Qm_stacked_RGB[j], c=color, linewidth=2)  # Thicker lines
            plt.plot(self.Qp_stacked_RGB[0], self.Qp_stacked_RGB[j], c=color, linewidth=2, linestyle='--')  # Thicker lines
            plt.plot(self.Qm_stacked_RGB[0], self.Qm_stacked_RGB[j], c=color, linewidth=2, linestyle=':')  # Thicker lines

        plt.legend(["Red", "Green", "Blue",
                    "Red_Qm", "Green_Qm", "Blue_Qm",
                    "Red_Qp", "Green_Qp", "Blue_Qp"], loc='upper right', fontsize=10)
        plt.xlabel("Wavelength [nm]", fontsize=14, fontweight='bold')
        plt.ylabel("Intensity [a.u.]", fontsize=14, fontweight='bold')
        plt.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.ylim(0, np.nanmax(self.Qp_stacked_RGB[1:]+self.Qm_stacked_RGB[1:])*1.1)  # 10% more space above the max value
        plt.xlim(390, 700)

        if self.output_plots:
            plt.savefig(os.path.join(self.save_path, f"{self.label}_spectrum.png"), bbox_inches="tight", dpi=300)
        plt.close()

    def plot_bounding_boxes(self, **kwargs):
        """
        Plot the spectrum bounding boxes on top of the post-processed (RGB) raw image.
        """
        plt.imshow(self.img_post, **kwargs)
        plt.axvspan(self.slice_Qp.start, self.slice_Qp.stop, facecolor="white", edgecolor="white", alpha=0.1, ls="--")
        plt.axvspan(self.slice_Qm.start, self.slice_Qm.stop, facecolor="white", edgecolor="white", alpha=0.1, ls="--")
        plt.title(f"Bounding boxes\n{self.label}")
        plt.savefig(os.path.join(self.save_path, f"{self.label}_RGB_bounding_boxes.png"), bbox_inches="tight")
        plt.close()

    def find_spectrum_slices(self, model="SPIE"):
        """
        Find the x and y limits that contain the two spectra in the image
        FIXME: Hardcoded for now
        """
        if (model is None) or (model == "SPIE"):
            # SPIE paper defaults
            slice_Qp = np.s_[650:1400]
            slice_Qm = np.s_[1550:2300]
        return slice_Qp, slice_Qm
    
    def _gauss_nan(self, D, sigma=5, **kwargs):
        """
        Apply a multidimensional Gaussian kernel, accounting for NaN values.
        Reference: https://stackoverflow.com/a/36307291/2229219
        """
        V = D.copy()
        V[D!=D] = 0
        VV = gaussMd(V, sigma=sigma, **kwargs)

        W = 0 * D.copy() + 1
        W[D!=D] = 0
        WW = gaussMd(W, sigma=sigma, **kwargs)

        Z=VV/WW
        return Z
    
    def _generate_bayer_slices(self, color_pattern, colours=range(4)):
        """
        Generate the slices used to demosaick data.
        """
        # Find the positions of the first element corresponding to each colour
        positions = [np.array(np.where(color_pattern == colour)).T[0] for colour in colours]

        # Make a slice for each colour
        slices = [np.s_[..., x::2, y::2] for x, y in positions]

        return slices


    def demosaick(self, bayer_map, data, color_desc="RGBG"):
        """
        Uses a Bayer map `bayer_map` (RGBG channel for each pixel) and any number
        of input arrays `data`.
        """
        # Cast the data to a numpy array for the following indexing tricks to work
        data = np.array(data)

        # Check that we are dealing with RGBG2 data, as only these are supported right now.
        assert color_desc in ("RGBG", b"RGBG"), f"Unknown colour description `{color_desc}"

        # Check that the data and Bayer pattern have similar shapes
        assert data.shape[-2:] == bayer_map.shape, f"The data ({data.shape}) and Bayer map ({bayer_map.shape}) have incompatible shapes"

        # Demosaick the data along their last two axes
        bayer_pattern = bayer_map[:2, :2]
        slices = self._generate_bayer_slices(bayer_pattern)

        # Combine the data back into one array of shape [..., 4, x/2, y/2]
        newshape = list(data.shape[:-2]) + [4, data.shape[-2]//2, data.shape[-1]//2]
        RGBG = np.empty(newshape)
        for i, s in enumerate(slices):
            RGBG[..., i, :, :] = data[s]

        return RGBG
    
    def interpolate(self, wavelength_array, color_value_array, lambdarange):
        interpolated = np.array([np.interp(lambdarange, wavelengths, color_values) for wavelengths, color_values in zip(wavelength_array, color_value_array)])
        return interpolated

    def interpolate_multi(self, wavelengths_split, RGBG, lambdamin=390, lambdamax=700, lambdastep=1):
        lambdarange = np.arange(lambdamin, lambdamax+lambdastep, lambdastep)
        all_interpolated = np.array([self.interpolate(wavelengths_split[c], RGBG[c], lambdarange) for c in range(4)])
        all_interpolated = np.moveaxis(all_interpolated, 2, 1)
        return lambdarange, all_interpolated

    def stack(self, wavelengths, interpolated):
        """
        Combine two Green channels and add wavelength grid
        Outputs [WL, R, G, B] array
        """
        stacked = interpolated.mean(axis=2)
        stacked = np.roll(stacked, 1, axis=0)      # move to make space for wavelengths
        stacked[2] = (stacked[0] + stacked[2])/2.  # G becomes mean of G
        stacked[0] = wavelengths  # put wavelengths into array
        return stacked