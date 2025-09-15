#! /usr/bin/env python
import logging
from classes import Ispeximage
import glob
import os
import re

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('ispex')

# where are the images stored
img_path = os.path.abspath("example_data/iSPEX_Set_20250806_0925_3537")
save_path = os.path.abspath(os.path.join("example_outputs", os.path.basename(img_path)))

if not os.path.isdir(save_path):
    os.makedirs(save_path)

# find all the images
if not os.path.exists(img_path):
    log.error(f"File not found: {img_path}")
    raise FileNotFoundError(f"File path not found: {img_path}")

images = glob.glob(os.path.join(img_path, '*.DNG'))
if len(images) == 0:
    raise FileNotFoundError(f"No images found in {img_path}")

# dictionary to organise the files
card_set = {'E0': None, 'E1': None, 'E2': None, 'E3': None, 'E4': None}
water_set = {'E0': None, 'E1': None, 'E2': None, 'E3': None, 'E4': None}
sky_set = {'E0': None, 'E1': None, 'E2': None, 'E3': None, 'E4': None}

# Check that file spec conforms to expected pattern and populate the dictionary
for impath in images:
    log.info(f"Image: {impath}")
    pattern = re.compile(r"IMG_(?P<datestr>\d{8})_(?P<timestr>\d{4})_(?P<uuid>\w{4})_(?P<obstype>\w{1})_(?P<exposure_seq>\w{2}).DNG")
    match = pattern.match(os.path.basename(impath))
    if match is None:
        raise ValueError(f"Filename {os.path.basename(impath)} does not match expected pattern.")

    obstype = match.groupdict()['obstype']  # 'C', 'S', or 'W' for Card, Sky, Water
    exposure = match.groupdict()['exposure_seq']  # E0, E1, E2, E3, or E4
    # below, we override the cal set lookup (pick E2) until we have a way to select the best exposure
    if obstype == 'C':
        card_set[exposure] = Ispeximage(dng_path=impath,
                                        type='observation',
                                        save_path_root='example_outputs/iSPEX_Set_20250806_0925_3537',
                                        calibration_set='cameras/iPhone14_4/20250813_1501_59E5_E2')
    elif obstype == 'W':
        water_set[exposure] = Ispeximage(dng_path=impath,
                                         type='observation',
                                         save_path_root='example_outputs/iSPEX_Set_20250806_0925_3537',
                                        calibration_set='cameras/iPhone14_4/20250813_1501_59E5_E2')
    elif obstype == 'S':
        sky_set[exposure] = Ispeximage(dng_path=impath,
                                       type='observation',
                                       save_path_root='example_outputs/iSPEX_Set_20250806_0925_3537',
                                        calibration_set='cameras/iPhone14_4/20250813_1501_59E5_E2')
    else:
        log.warning(f"Unknown observation type {obstype} in file {impath}")

# Process each set
for set in [card_set, water_set, sky_set]:
    for exposure in set:
        if set[exposure] is None:
            log.warning(f"Missing image for exposure {exposure}")
        else:
            try:
                log.info(f"Processing {exposure}")
                set[exposure].process()
                set[exposure].plot_bounding_areas()
                #set[exposure].plot_background_correction()
                set[exposure].plot_spectra()
            except Exception as e:
                log.error(f"Error processing {set[exposure].dng_path}: {e}")
                continue

breakpoint()

# quality control - which image exposures should be used for Rrs
# Produce Rrs (using new class in classes.py)