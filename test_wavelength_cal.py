#! /usr/bin/env python
import os
from classes import Ispeximage
import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger('ispex')

dng_path = './example_data/amt352/352/cal/IMG_0669.DNG'

tl_cal = Ispeximage(dng_path=dng_path,
                    save_path=os.path.dirname(dng_path),
                    camera='apple_iphone_mini_13',
                    output_plots=False,
                    type='fluorescent_lamp_cal'
                    )
