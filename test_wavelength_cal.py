#!/usr/bin/env python
"""
Process iSPEX calibration images of fluorescent lamp
Only using the 'C' (card) observations, but looking at each Exposure
"""

import os
from classes import Ispeximage
import logging
import glob
import re

# set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger('ispex')

# where are the images stored
cal_path = os.path.abspath('example_data/wavelength-cals/iSPEX_Set_20250813_1501_59E5')

# find all the calibration images
if not os.path.exists(cal_path):
    log.error(f"File not found: {cal_path}")
    raise FileNotFoundError(f"File path not found: {cal_path}")

cal_images = glob.glob(os.path.join(cal_path, '*.DNG'))
if len(cal_images) == 0:
    raise FileNotFoundError(f"No images found in {cal_path}")

# dictionary to organise the files
cal_set = {'E0': None, 'E1': None, 'E2': None, 'E3': None, 'E4': None}

# Check that file spec conforms to expected pattern and populate the dictionary
for impath in cal_images:
    log.info(f"Processing calibration image: {impath}")
    pattern = re.compile(r"IMG_(?P<datestr>\d{8})_(?P<timestr>\d{4})_(?P<uuid>\w{4})_(?P<obstype>\w{1})_(?P<exposure_seq>\w{2}).DNG")
    match = pattern.match(os.path.basename(impath))
    if match is None:
        raise ValueError(f"Filename {os.path.basename(impath)} does not match expected pattern.")
    obstype = match.groupdict()['obstype']  # 'C', 'S', or 'W' for Card, Sky, Water
    if obstype != 'C':
        log.info(f"Skipping non-card image: {impath}")
        continue
    exposure = match.groupdict()['exposure_seq']  # E0, E1, E2, E3, or E4
    cal_set[exposure] = Ispeximage(dng_path=impath,
                                   type='fluorescent_lamp_cal',
                                   save_path_root='example_data/wavelength-cals/iSPEX_Set_20250813_1501_59E5',
                                   calibration_root='cameras')

# Process each image in the calibration set
for exposure in cal_set:
    if cal_set[exposure] is None:
        log.warning(f"Missing image for exposure {exposure}")
    else:
        log.info(f"Processing exposure {exposure}")
        cal_set[exposure].process()
        cal_set[exposure].plot_bounding_areas()
        # self.plot_background_correction()
        # self.plot_spectra()
        # self.plot_fluorescent_lines()

