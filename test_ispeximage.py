#! /usr/bin/env python
import logging
from classes import Ispeximage
import glob
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('ispex')

#dng_path='./example_data/card/1b7cdf9729b23e3b81bc39efaec9b2c4_rawFile.dng'
#dng_path = './example_data/amt352/352/Set1/IMG_0642.DNG'

dng_paths = glob.glob("./example_data/amt35[06]/*.DNG")
for dng_path in dng_paths:
    log.info(f'Processing {dng_path}')
    # Create an Ispeximage object for each DNG file
    card = Ispeximage(dng_path=dng_path,
                    save_path=os.path.dirname(dng_path),
                    output_plots=True,
                    type='observation')
