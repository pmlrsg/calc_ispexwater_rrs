import os
import numpy as np
from matplotlib import pyplot as plt
import rawpy as rawpy_lib
from scipy.ndimage import gaussian_filter as gaussMd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
import logging


class Ispeximage(object):
    """
    An instance of Ispeximage class is created for each image file.
    The class contains methods to process the image file and save the processed data.
    """
    def __init__(self,
                 dng_path=None,
                 save_path=None,
                 camera='apple_iphone_mini_13',
                 output_plots=False,
                 type='observation'):
        """
        Initialize the Ispeximage object with the path to the dng image file,
          camera profile, and output directory (if producing plots).

        param dng_path: str, path to the dng image file
        param save_path: str, path to the output directory
        param camera: str, name of camera profile
        param output_plots: bool, whether to produce plots
        param type: str, one of ['observation', 'fluorescent_lamp_cal']
        """
        self.log = logging.getLogger('ispex.image')

        self.dng_path = dng_path            #  source RAW image file
        self.label = os.path.basename(dng_path).split(".")[0]
        self.save_path = save_path          #  output directory for plots, data files
        self.output_plots = output_plots    #  whether to produce plots
        self.type = type                    #  type of image, used to determine processing steps

        self.img_raw = None                 #  raw image data
        self.img_post = None                #  raw image data mapped to RGB, for visualisation purposes only
        self.camera = camera                #  camera model, used to load calibration profile
        self.img_raw_RGBG = None            #  raw image demosaicked to RGBG space
        self.img_raw_RGB = None             #  raw image mapped to RGB (0-1 scale), for visualisation purposes only

        # Quality Control                   
        self.check_areas = False            #  If False, the slit and/or projected areas could not be found

        # Masks (0/1 encoded)
        self.slit_area_mask = None          #  mask for the slit area
        self.projected_area_mask = None     #  mask for the projected area   
        # Edges along the slit dimension
        self.start_qm = None                #  index of the start of the Qm area
        self.end_qm = None                  #  index of the end of the Qm area
        self.start_qp = None                #  index of the start of the Qp area
        self.end_qp = None                  #  index of the end of the Qp area
        # Top and bottom edges of the projected area k-means cluster 
        self.top_qx = None                  #  index of the top of the projected area
        self.bottom_qx = None               #  index of the bottom of the projected area

        self.slice_Qm = None                # Qx pixel selection window
        self.slice_Qp = None                # 

        self.Qm_sliced = None               # Qx raw data block
        self.Qp_sliced = None               # 

        self.Qm_background_corrected = None # Qx raw data block, background corrected
        self.Qp_background_corrected = None # 
        self.background = None              # Background noise level interpolation, in brightness units

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
        Process the image file in memory.
        Optionally write out plots.
        """
        self.log = logging.getLogger('ispex.image.process')
        self.log.info(f"Reading image {self.label}")
        # read the raw image
        with rawpy_lib.imread(self.dng_path) as img:
            # Ensure image type is as expected (RawType.Flat)
            try:
                assert img.raw_type.name == "Flat"
            except AssertionError:
                raise(f"Invalid raw type: {img.raw_type}. Expected RawType.Flat")

            # obtain the raw image and bayer RGBG pixel mapping pattern
            self.img_raw = img.raw_image.astype(np.int16)  # was np.float64
            bayer_map = img.raw_colors

            #  scaled rgb image for visualisation, on the bayer pattern (interleaved RGBG pixels)
            #  not recommended for analysis of any kind.
            self.img_post = img.postprocess()

        # Ensure images always have the same orientation
        # Rotate if if the vertical dimension is longer than the horizontal
        if self.img_raw.shape[0] > self.img_raw.shape[1]:
            self.img_raw = np.rot90(self.img_raw)
        if bayer_map.shape[0] > bayer_map.shape[1]:
            bayer_map = np.rot90(bayer_map)
        if self.img_post.shape[0] > self.img_post.shape[1]:
            self.img_post = np.rot90(self.img_post)


        # Demosaick RAW image to RGBG format using the bayer pattern
        self.img_raw_RGBG = self.demosaick(bayer_map, self.img_raw)
        # combine G channels and normalise to 0-1 scale for visualisation
        self.img_raw_RGB = self._raw2RGB(self.img_raw_RGBG)


        self.log.info(f"Identify slit and projected image areas")
        self.find_areas()
        # self.plot_bounding_boxes()

        # Background correction
        self.log.info(f"Interpolate background brightness")
        # FIXME: do this on RGB instead of RGBG to save time
        self.background_solver()
        # self.plot_background_correction()
        self.img_bg_corrected = self.img_raw_RGBG - self.background

        if self.type == 'fluorescent_lamp_cal':
            self.process_fluorescent_lamp_calibration()


    # NOTE TO FUTURE SELF: resolve external dependencies in the following function.
    # Then produce the calibration information needed to work in the RGBG space.
    def process_fluorescent_lamp_calibration(self):
        """
        Determine wavelength calibration from a fluorescent lamp calibration image.
        These images should be obtained in a dark environment with a fluorescent lamp
        as the only light source illuminating a spectrally neutral panel diffuse panel.
        """

        self.log.info(f"Processing fluorescent lamp calibration image")

        # Convert the RGB image to summed intensity
        # img_grey = np.dot(self.img_post[..., :3], [0.33, 0.33, 0.33])
        img_grey = np.nansum(self.img_raw_RGB, axis=2)

        # Sum along the spectrum axis to find peaks corresponding to lamp 
        along_projection_sum = np.sum(img_grey, axis=0)

        # Ignore the part of the image that thas the slit
        right_side_data = along_projection_sum[int(self.top_qx*0.8):]

        # Set a threshold to find the significant peaks
        threshold = 0.3 * np.max(right_side_data)
        
        # Find peaks with the specified threshold
        peaks = self._find_cal_peaks(right_side_data, threshold=threshold)

        # Adjust the peaks to account for the midpoint offset
        adjusted_peaks = [peak + int(self.top_qx*0.8) for peak in peaks]
        spectrum_start_pixel = min(adjusted_peaks)


        # # crude bias calibration from rawpy, i.e. without SPECTACLE calibrations
        # self.img_raw_bc = self.img_raw - float(img.black_level_per_channel[0])

        # #define start and end position for the slices containing slit + spectrum for Qm/Qp in the image (based on hardcoded iPhoneSE data)
        # slice_Qp, slice_Qm = ispex_general.find_spectrum(data)
        # # Show slices on top of the original image (probably to check if it's aligned correctly)
        # # ispex_plot.plot_bounding_boxes(img_post, label_file=file, saveto="bounding_boxes.pdf")

        # # cut out 2 slices of 750 pixels high x 4032 pixels wide on the data and bayer map images
        # data_Qp, data_Qm = data[slice_Qp], data[slice_Qm]
        # bayer_Qp, bayer_Qm = bayer_map[slice_Qp], bayer_map[slice_Qm]

        # #Debayer the 4 channels RGBG RAW image into RGB data
        # RGB_Qp = raw2.pull_apart2(data_Qp, bayer_Qp)
        # RGB_Qm = raw2.pull_apart2(data_Qm, bayer_Qm)

        #Define variables for size 
        # x = np.arange(data_Qp.shape[1])
        # yp = np.arange(data_Qp.shape[0])
        # ym = np.arange(data_Qm.shape[0])

        # #add extra xp and xm for raw_demosaic at bottom of script
        # xp = np.repeat(x[:,np.newaxis], bayer_Qp.shape[0], axis=1).T
        # xm = np.repeat(x[:,np.newaxis], bayer_Qm.shape[0], axis=1).T

        # # Convolve the data with a Gaussian kernel on the wavelength axis to remove noise
        # TODO: is this strictly necessary at this point in processing?
        breakpoint() 
        gauss_Qp = general.gauss_filter_multidimensional(RGB_Qp, sigma=(0,0,6))
        gauss_Qm = general.gauss_filter_multidimensional(RGB_Qm, sigma=(0,0,6))

        # Find the range of pixel values for the R,G,B peaks in the image
        # TODO: look at this function to see whether this could take the known spectrum slice as input directly rather than the 'spectrum_start_pixel'
        lines_Qp = _find_fluorescent_lines(gauss_Qp[...,spectrum_start_pixel:]) + spectrum_start_pixel
        lines_Qm = _find_fluorescent_lines(gauss_Qm[...,spectrum_start_pixel:]) + spectrum_start_pixel

        lines_fit_Qp = _fit_fluorescent_lines(lines_Qp, yp)
        lines_fit_Qm = _fit_fluorescent_lines(lines_Qm, ym)

        ispex_plot.plot_fluorescent_lines(yp, lines_Qp, lines_fit_Qp,saveto="fl_linesQp.pdf")
        ispex_plot.plot_fluorescent_lines(ym, lines_Qm, lines_fit_Qm,saveto="fl_linesQm.pdf")

        ispex_plot.plot_fluorescent_lines_double([yp, ym], [lines_Qp, lines_Qm], [lines_fit_Qp, lines_fit_Qm], saveto="TL_calibration.pdf")

        # Calculate the dispersion (nm/pixel) for each row (R line - B line) / (R pixel - B pixel)
        dispersion_Qp = wvl.dispersion_fluorescent(lines_fit_Qp)
        dispersion_Qm = wvl.dispersion_fluorescent(lines_fit_Qm)

        #plot the lines, the fit and the dispersion and save to file
        ispex_plot.plot_fluorescent_lines_dispersion([yp, ym], [lines_Qp, lines_Qm], [lines_fit_Qp, lines_fit_Qm], [dispersion_Qp, dispersion_Qm], saveto="TL_calibration_dispersion.pdf")

        # Calculate the spectral resolution for all rows
        resolution_Qp = wvl.resolution(gauss_Qp, dispersion_Qp)
        resolution_Qm = wvl.resolution(gauss_Qm, dispersion_Qm)
        # print(f"Resolution Qp: {resolution_Qp}")
        # print(f"Resolution Qm: {resolution_Qm}")

        # Fit a wavelength relation for each row, meaning: try to fit a polynomial to the 3 lines (R, G, B) with 3 coefficients
        # Using an ax^2 + bx + c function with the coefficients to match the wavelength, where x = the pixel value that corresponds with R,G,B
        wavelength_fits_Qp = wavelength.fit_many_wavelength_relations(yp, lines_fit_Qp)
        wavelength_fits_Qm = wavelength.fit_many_wavelength_relations(ym, lines_fit_Qm)

        # Fit a polynomial to the coefficients of the previous fit
        #These values are the most important of the whole script, as this array of 15 values can be used
        # to calculate any value of wavelength for any pixel in the image!
        coefficients_Qp, coefficients_fit_Qp = wavelength.fit_wavelength_coefficients(yp, wavelength_fits_Qp)
        coefficients_Qm, coefficients_fit_Qm = wavelength.fit_wavelength_coefficients(ym, wavelength_fits_Qm)
        # print(coefficients_Qp)

        # Save the coefficients to file for use with other scripts like spectrum.py
        wavelength.save_coefficients(coefficients_Qp, saveto=save_to_Qp)
        wavelength.save_coefficients(coefficients_Qm, saveto=save_to_Qm)
        print(f"Saved wavelength coefficients to '{save_to_Qp}' and '{save_to_Qm}'")

        # Convert the input image pixel values to wavelengths values using the coefficients
        wavelengths_Qp = wavelength.calculate_wavelengths(coefficients_Qp, x, yp)
        wavelengths_Qm = wavelength.calculate_wavelengths(coefficients_Qm, x, ym)

        # Demoisaic the image by splitting the image into 4 channels (R, G, B, G) and interpolating the pixel intensities for the bayer pattern 
        #this halves the width and height of the image, so the image is now 750 pixels high x 2016 pixels wide
        wavelengths_split_Qp, RGBG_Qp, xp_split = ispex_raw.demosaick(bayer_Qp, [wavelengths_Qp, data_Qp, xp])
        wavelengths_split_Qm, RGBG_Qm, xm_split = ispex_raw.demosaick(bayer_Qm,[wavelengths_Qm, data_Qm, xm])

        #Extra smoothing on the curve (OPTIONAL)
        RGBG_Qp = general._gauss_nan(RGBG_Qp, sigma=(0,0,3))
        RGBG_Qm = general._gauss_nan(RGBG_Qm, sigma=(0,0,3))

        #Interpolate all float values pixel values that contain the wavelength of that pixel to the lambdarange (390-700 nm) with a step of 1 nm
        lambdarange, all_interpolated_Qp = wavelength.interpolate_multi(wavelengths_split_Qp, RGBG_Qp)
        lambdarange, all_interpolated_Qm = wavelength.interpolate_multi(wavelengths_split_Qm, RGBG_Qm)

        #Stack and plot the spectrum
        stacked_Qp = wavelength.stack(lambdarange, all_interpolated_Qp)
        stacked_Qm = wavelength.stack(lambdarange, all_interpolated_Qm)

        plot.plot_fluorescent_spectrum(stacked_Qp[0], stacked_Qp[1:])
        plot.plot_fluorescent_spectrum(stacked_Qm[0], stacked_Qm[1:])




        # OLD CODE
        # raw and demosaicked image have the short axis (along-slit) mirrored for some reason. x-y order also swapped.
        slice_Qp = np.s_[650:1400]
        slice_Qm = np.s_[1550:2300]
        data_Qp, data_Qm = self.img_raw[slice_Qp], self.img_raw[slice_Qm]
        bayer_Qp, bayer_Qm = bayer_map[slice_Qp], bayer_map[slice_Qm]
        x = np.arange(data_Qp.shape[1])                                 #  (4032,)
        xp = np.repeat(x[:,np.newaxis], bayer_Qp.shape[0], axis=1).T    #  (750, 4032) (width of Q slice, length of image)
        xm = np.repeat(x[:,np.newaxis], bayer_Qm.shape[0], axis=1).T    #  (750, 4032) (width of Q slice, length of image)
        yp = np.arange(data_Qp.shape[0])                                #  (750,)
        ym = np.arange(data_Qm.shape[0])                                #  (750,)
        #wavelengths_Qp = self.calculate_wavelengths(self.wl_calib_qp, x, yp)
        #wavelengths_Qm = self.calculate_wavelengths(self.wl_calib_qm, x, ym)
        coeff_fit = np.array([np.polyval(c, yp) for c in self.wl_calib_qp]).T       # (750,3)
        wavelengths_Qp = np.array([np.polyval(c_fit, x) for c_fit in coeff_fit])    # (750, 4032) ranging -954.2 to 787.616 in first row, -22147.98 to -2328.61 in last row
        coeff_fit = np.array([np.polyval(c, ym) for c in self.wl_calib_qm]).T       # (750,3)
        wavelengths_Qm = np.array([np.polyval(c_fit, x) for c_fit in coeff_fit])    # (750, 4032) ranging -536.9 to 850.7 in first row, -975.2 to 731.5 in last row
        
        # updated code
        bayer_Qp = bayer_map[np.s_[self.start_qp:self.end_qp]][:,::2]               #  (373, 2016)
        bayer_Qm = bayer_map[np.s_[self.start_qm:self.end_qm]][:,::2]               #  (393, 2016)
        x = np.arange(self.img_raw_RGBG.shape[2])                                   #  (2016,)
        xp = np.repeat(x[:,np.newaxis], bayer_Qp.shape[0], axis=1).T                #  (373, 2016) (width of Q slice, length of image)
        xm = np.repeat(x[:,np.newaxis], bayer_Qm.shape[0], axis=1).T                #  (393, 2016) (width of Q slice, length of image)
        yp = np.arange(self.end_qp-self.start_qp)                                   #  (373,)
        ym = np.arange(self.end_qm-self.start_qm)                                   #  (393,)
        coeff_fit = np.array([np.polyval(c, yp) for c in self.wl_calib_qp]).T       #  (373,3)
        wavelengths_Qp = np.array([np.polyval(c_fit, x) for c_fit in coeff_fit])    #  (373, 2016) ranging -954.2 to 787.616 in first row, -22147.98 to -2328.61 in last row
        coeff_fit = np.array([np.polyval(c, ym) for c in self.wl_calib_qm]).T       #  (393,3)
        wavelengths_Qm = np.array([np.polyval(c_fit, x) for c_fit in coeff_fit])    #  (750, 2016) ranging -536.9 to 850.7 in first row, -975.2 to 731.5 in last row
        breakpoint()
        wavelengths_RGBG_Qp = self.demosaick(bayer_Qp, wavelengths_Qp)
        wavelengths_RGBG_Qp = self.demosaick(bayer_Qp, wavelengths_Qm)
        
        #wavelengths_split_Qp, RGBG_Qp, xp_split = self.demosaick(bayer_Qp, [wavelengths_Qp, data_Qp, xp])  # wavelengths, RGBQ_Qx split: (4, 375, 2016)
        #wavelengths_split_Qm, RGBG_Qm, xm_split = self.demosaick(bayer_Qm, [wavelengths_Qm, data_Qm, xm]) 
        breakpoint()
        
        # Updated code
        # Map pixels to wavelength grid
        buffer = 0.10  #  % buffer along the slit dimension to avoid edge effects
        x = np.arange(self.img_bg_corrected.shape[2])                           #  along-spectrum axis length (shorter to longer wavelength), same for Qp and Qm
        xp = np.repeat(x[:,np.newaxis], self.end_qm - self.start_qm, axis=1).T  #  along-spectrum axis repeated for each line in Qp
        xm = np.repeat(x[:,np.newaxis], self.end_qp - self.start_qp, axis=1).T  #  along-spectrum axis repeated for each line in Qm
        yp = np.arange(self.end_qp - self.start_qp)  # along-slit axis of length of Qp section
        ym = np.arange(self.end_qm - self.start_qm)  # along-slit axis of length of Qm section
        # apply the polynomial fit of the wavelength calibration to the pixel coordinates
        coeff_fit = np.array([np.polyval(c, yp) for c in self.wl_calib_qp]).T
        wavelengths_Qp = np.array([np.polyval(c_fit, x) for c_fit in coeff_fit])
        coeff_fit = np.array([np.polyval(c, ym) for c in self.wl_calib_qm]).T
        wavelengths_Qm = np.array([np.polyval(c_fit, x) for c_fit in coeff_fit])
        breakpoint()

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
        
        # stack the RGBG into a WL,R,G,B spectra (4xN) 
        self.Qp_stacked_RGB = self.stack(lambdarange, Qp_RGBG)  # RGB radiance in arbitrary units
        self.Qm_stacked_RGB = self.stack(lambdarange, Qm_RGBG)  # RGB radiance in arbitrary units

        if self.output_plots:
            self.plot_bounding_boxes()
            self.plot_spectra()

    def find_areas(self):
        """
        Find the slit and projected areas in the image
        """
        cut_tolerance = 0.05  # % of max value in projected areas is used to slice the image

        img_raw_sum = np.nansum(self.img_raw_RGB, axis=2)
        img_raw_sum_1d = img_raw_sum.flatten().reshape(-1, 1)

        # Use KMeans clustering (2 clusters) to find and mask slit
        # Slit area should be 1-2 orders of magnitude brighter than projected spectral area so should separate easily
        kmeans = KMeans(n_clusters=2, random_state=0).fit(img_raw_sum_1d)
        self.slit_area_mask = kmeans.labels_.reshape(img_raw_sum.shape)

        # Cluster the remainder of the image again to find te projected area
        # Ignore slit area by cutting the image with a 20% buffer
        buffer_width = int(np.floor(np.min(img_raw_sum.shape)* 0.2))  # 20% buffer of image width
        max_index = np.max(np.where(self.slit_area_mask == 1)[1])
        cut_index = max_index + buffer_width

        img_raw_sum_bottom_half = img_raw_sum[:, cut_index:]
        img_raw_sum_bottom_half_1d = img_raw_sum_bottom_half.flatten().reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(img_raw_sum_bottom_half_1d)
        labels = kmeans.labels_.reshape(img_raw_sum_bottom_half.shape)
        self.projected_area_mask = np.zeros_like(self.slit_area_mask)
        self.projected_area_mask[:, cut_index:] = labels

        # aggregate the image along the slit dimension to find the two projected sections
        img_raw_sum_along_slit = img_raw_sum.copy()
        img_raw_sum_along_slit[self.projected_area_mask == 0] = 0
        img_raw_sum_along_slit = np.nansum(img_raw_sum_along_slit, axis=1)
        # cumulative sum along the slit dimension
        img_raw_sum_along_slit_cumsum = np.cumsum(img_raw_sum_along_slit)
        try:
            # top and bottom edges of the projected area k-means cluster 
            self.top_qx = np.argwhere(np.nansum(self.projected_area_mask, axis = 0) > 0)[0][0]
            self.bottom_qx = np.argwhere(np.nansum(self.projected_area_mask, axis = 0) > 0)[-1][0]

            # Find edges along the slit dimension
            cut_in = cut_tolerance
            cut_off = 1.0 - cut_tolerance
            # define the start of qp and end of qm from the cutting range of the cumulative sum
            maxsum = np.max(img_raw_sum_along_slit_cumsum)
            self.start_qp = np.where(img_raw_sum_along_slit_cumsum > cut_in*maxsum)[0][0]
            self.end_qm = np.where(img_raw_sum_along_slit_cumsum < cut_off*maxsum)[0][-1]

            # define the middle plateau (between qm and qp projections) as value where diff is lowest
            mid_plateau_start_index = self.start_qp + np.argmin(np.diff(img_raw_sum_along_slit_cumsum[self.start_qp:self.end_qm]))
            mid_plateau_start_value = img_raw_sum_along_slit_cumsum[mid_plateau_start_index]

            # define the end of qp where the plateau value is reached within set % of max sum
            self.end_qp = self.start_qp + \
                          np.where(img_raw_sum_along_slit_cumsum[self.start_qp:]
                                   < (mid_plateau_start_value - (cut_in*maxsum)))[0][-1]
            # define the start of qm where the plateau value is exceeded by  set % of max sum
            self.start_qm = mid_plateau_start_index + \
                          np.where(img_raw_sum_along_slit_cumsum[mid_plateau_start_index:] 
                                   > (mid_plateau_start_value + (cut_in*maxsum)))[0][0]

            self.check_areas = True
            assert self.start_qp < self.start_qm
            assert self.end_qm > self.start_qm
            assert self.end_qp > self.start_qp
            assert self.bottom_qx > self.top_qx

        except IndexError:
            self.log.error("Error finding projected areas (areas are invalid)")
            self.check_areas = False
        except AssertionError:
            self.log.error("Error finding projected areas (areas are too small)")
            self.check_areas = False
        except Exception:
            self.check_areas = False
            raise

    def background_solver(self):
        """
        Solve for the background noise level by interpolating across the RGB image layers.
        Note that G layers should already be combined into one.
        """
        # find image index halfway between slit and projected area
        try:
            end_of_slit_y = np.argwhere(self.slit_area_mask == 1)[-1][-1]
            start_of_proj_y = np.argwhere(self.projected_area_mask == 1)[-1][0]
            end_of_proj_y = np.argwhere(self.projected_area_mask == 1)[-1][-1]
            assert end_of_slit_y < start_of_proj_y
            assert end_of_proj_y > start_of_proj_y
            assert end_of_proj_y < self.img_raw_RGB.shape[1]
            background_slice_start = end_of_slit_y + (start_of_proj_y - end_of_slit_y) // 2
        except AssertionError:
            raise(Exception("Error finding background area (areas are invalid)"))

        self.background = np.zeros_like(self.img_raw_RGB)
        background_slice = self.img_raw_RGB.copy()
        # mask the slit area
        background_slice[:, :background_slice_start, :] = np.nan
        # # mask the projected areas with 20% buffer
        # mask the projected area as one large symmetrical rectangle based on buffered projection bounds
        # define symmetrical bounds for the projected area
        long_edge_to_proj = np.min([self.start_qp, self.img_raw_RGB.shape[0] - self.end_qm])
        slice_x_start = int(long_edge_to_proj * 0.8)
        slice_x_end = int(self.img_raw_RGB.shape[0] - slice_x_start)
        slice_y_start = int(start_of_proj_y * 0.8)
        slice_y_end = int(end_of_proj_y * 1.2)
        try:
            assert slice_x_end < self.img_raw_RGB.shape[0]
            assert slice_y_end < self.img_raw_RGB.shape[1]
        except AssertionError:
            raise(Exception("Error finding projected areas (buffered area bounds exceed image bounds)"))
        background_slice[slice_x_start:slice_x_end, slice_y_start:slice_y_end, :] = np.nan
        breakpoint()
        uncertainty_background_2sigma = {}

        for i, layername in enumerate(['R', 'G', 'B']):
            layer = background_slice[i]
            if np.isnan(layer).all():
                continue

            X = np.array(np.meshgrid(np.arange(layer.shape[0]),
                                     np.arange(layer.shape[1]))).T.reshape(-1, 2)
            y = layer.flatten()
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            if len(y) == 0:
                continue

            self.log.info(f"Interpolating background noise level for layer {i}")
            regressor = HistGradientBoostingRegressor()
            regressor.fit(X, y)
            # background_slice[i] = regressor.predict(np.arange(layer.shape[0]).reshape(-1, 1))

            background_pred = regressor.predict(np.array(np.meshgrid(np.arange(layer.shape[0]),
                                                                     np.arange(layer.shape[1]))).T.reshape(-1, 2))
            layer_interpolated = background_pred.reshape(layer.shape)

            layer_interp_qm = layer_interpolated[int(self.start_qm * 0.8):int(self.end_qm * 1.2), int(self.top_qx * 0.8):int(self.bottom_qx * 1.2)]
            layer_interp_qp = layer_interpolated[int(self.start_qp * 0.8):int(self.end_qp * 1.2), int(self.top_qx * 0.8):int(self.bottom_qx * 1.2)]
            layer_original_qm = layer[int(self.start_qm * 0.8):int(self.end_qm * 1.2), int(self.top_qx * 0.8):int(self.bottom_qx * 1.2)]
            layer_original_qp = layer[int(self.start_qp * 0.8):int(self.end_qp * 1.2), int(self.top_qx * 0.8):int(self.bottom_qx * 1.2)]
            layer_interp_qx = np.concat([layer_interp_qm, layer_interp_qp], axis=0)
            layer_original_qx = np.concat([layer_original_qm, layer_original_qp], axis=0)
            uncertainty_background_2sigma[layername] = np.std(layer_interp_qx - layer_original_qx) * 2.0
            
            # Replace the NaN values in the original background_slice with the interpolated values
            layer[np.isnan(layer)] = layer_interpolated[np.isnan(layer)]
            self.background[i] = layer

    def plot_bounding_boxes(self):
        """
        Plot the spectrum bounding boxes on top of the post-processed (RGB) raw image.
        """
        # normalise raw to RGB, boost values
        plt.imshow(self.img_raw_RGB)
        plt.contour(self.slit_area_mask, levels=[0.1], colors='magenta', linewidths=1.5)
        plt.contour(self.projected_area_mask, levels=[0.1], colors='magenta', linewidths=0.5)

        plt.axhspan(self.start_qm, self.end_qm, facecolor="white", edgecolor="red", alpha=0.1, ls="--")
        plt.axhspan(self.start_qp, self.end_qp, facecolor="white", edgecolor="red", alpha=0.1, ls="--")
        plt.axvspan(self.top_qx, self.bottom_qx, facecolor="white", edgecolor="red", alpha=0.1, ls="--")

        plt.title(f"Bounding areas\n{self.label}")
        plt.text(s = "k-means clustering of slit and projected areas",
                 x = self.slit_area_mask.shape[1]*0.05,
                 y = self.slit_area_mask.shape[0]*0.90, color="magenta")
        plt.text(s = "White level boosted 50%, dark pixel subtracted",
                 x = self.slit_area_mask.shape[1]*0.05,
                 y = self.slit_area_mask.shape[0]*0.95, color="white")

        plt.text(s = 'Qm', x = self.bottom_qx * 1.02, y = self.start_qm + ((self.end_qm - self.start_qm) /2), color="red")
        plt.text(s = 'Qp', x = self.bottom_qx * 1.02, y = self.start_qp + ((self.end_qp - self.start_qp) /2), color="red")

        plt.savefig(os.path.join(self.save_path, f"{self.label}_RGB_bounding_areas.png"), bbox_inches="tight")
        plt.close()

    def plot_background_correction(self):
        """
        Plot the background correction for each layer
        """
        for i, layername in enumerate(['R', 'G0', 'B', 'G1']):
            relative_bg_correction = 100.0*(self.img_raw_RGBG[i]-self.background[i])/self.img_raw_RGBG[i]
            relative_bg_correction[:,:self.top_qx] = np.nan
            plt.imshow(relative_bg_correction, cmap='coolwarm', vmin=0)
            plt.colorbar()
            rel_bg_corr_mean = np.nanmean(relative_bg_correction[self.start_qm:self.end_qp, self.top_qx:self.bottom_qx])
            rel_bg_corr_std = np.nanstd(relative_bg_correction[self.start_qm:self.end_qp, self.top_qx:self.bottom_qx])
            plt.text(s=f"{layername} background correction\n {rel_bg_corr_mean:2.2f} +/- {rel_bg_corr_std:2.2f} %",
                    x=self.background[i].shape[1]*0.05,
                    y=self.background[i].shape[0]*0.95,
                    color="black")
            plt.savefig(os.path.join(self.save_path, f"{self.label}_background_{layername}.png"), bbox_inches="tight")
            plt.close()

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
    
    def _raw2RGB(self, img_raw_RGBG):
        """
        Combine G channels,
        lower the white level by 50%
        subtract dark pixel,
        just for visualisation.
        Outputs 0-1 scaled RGB stack for imshow
        """
        R0 = img_raw_RGBG[0]
        G0 = (img_raw_RGBG[1] + img_raw_RGBG[3]) / 2.0
        B0 = img_raw_RGBG[2]
        whitelevel = np.max([R0, G0, B0])
        R0[R0 < (0.5 * whitelevel)] *= 2.0
        G0[G0 < (0.5 * whitelevel)] *= 2.0
        B0[B0 < (0.5 * whitelevel)] *= 2.0
        maxlevel = np.max([R0, G0, B0])
        minlevel = np.min([R0, G0, B0])
        R = (R0 - minlevel) / (maxlevel - minlevel)
        G = (G0 - minlevel) / (maxlevel - minlevel)
        B = (B0 - minlevel) / (maxlevel - minlevel)

        return np.dstack([R, G, B])

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
        assert color_desc in ("RGBG", b"RGBG"), f"Unknown colour description {color_desc}"

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
    
    def _find_cal_peaks(self, data: np.ndarray, threshold: float) -> list[int]:
        """
        Find peaks in a 1D array of data points above a certain threshold.

        Parameters:
        data (np.ndarray): The 1D array of data points.
        threshold (float): The threshold to identify significant peaks.

        Returns:
        List[int]: A list of indices where peaks are found.
        """
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold:
                peaks.append(i)
        return peaks
        
def _find_fluorescent_lines(RGB):
    RGB_copy = RGB.copy()
    RGB_copy[np.isnan(RGB_copy)] = -999
    peaks = np.nanargmax(RGB_copy, axis=2).astype(np.float32)
    peaks[peaks == 0] = np.nan
    return peaks

def _fit_fluorescent_lines(lines, y):
    lines_fit = lines.copy()
    for j in (0,1,2):  # fit separately for R, G, B
        # Filter out non-finite and NaN elements
        idx = np.isfinite(lines[j])
        new_y = y[idx] ; new_line = lines[j][idx]

        # Sigma-clip to filter out elements more than 3-sigma away from the mean
        clipped = sigma_clip(new_line)  # generates a masked array
        idx = ~clipped.mask  # get the non-masked items
        new_y = new_y[idx] ; new_line = new_line[idx]

        # Fit a polynomial to the line positions
        # Note: np.polyfit can go along axis - try this?
        coeff = np.polyfit(new_y, new_line, degree_of_spectral_line_fit)

        # Evaluate the fitted polynomial on all y positions
        lines_fit[j] = np.polyval(coeff, y)
    return lines_fit

