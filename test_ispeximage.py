#! /usr/bin/env python
import logging
from classes import Ispeximage

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('ispex')

card = Ispeximage(dng_path='./example_data/card/1b7cdf9729b23e3b81bc39efaec9b2c4_rawFile.dng',
                  save_path='.',
                  output_plots=False,
                  type='observation')
