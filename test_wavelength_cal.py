#!/usr/bin/env python
import os
from classes import Ispeximage
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger('ispex')

dng_path = os.path.abspath('example_data/lamp_cal/iSPEX_Set_57D0B619/IMG_57D0B619_C_E1.DNG')

if not os.path.exists(dng_path):
    log.error(f"File not found: {dng_path}")
    raise FileNotFoundError(f"File not found: {dng_path}")

tl_cal = Ispeximage(dng_path=dng_path,
                    save_path=os.path.dirname(dng_path),
                    camera='apple_iphone_mini_13',
                    output_plots=True,
                    type='fluorescent_lamp_cal'
                    )
