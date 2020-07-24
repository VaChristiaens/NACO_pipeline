#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:11:12 2020

@author: lewis
"""
#classification stage already completed

from naco_pip import input_dataset, raw_dataset

clas = input_dataset('/home/lpic0002/pd87_scratch/products/NACO_archive/5_HD179218/raw/',
                   '/home/lpic0002/pd87_scratch/products/NACO_archive/5_HD179218/classification/')
clas.bad_columns()
clas.mk_dico()
clas.find_sky_in_sci_cube(plot = 'save')

calib = raw_dataset('/home/lpic0002/pd87_scratch/products/NACO_archive/5_HD179218/classification/',
                   '/home/lpic0002/pd87_scratch/products/NACO_archive/5_HD179218/calib/')

calib.dark_subtract(debug = False, plot = False)
calib.flat_field_correction(debug = False, plot = False)
calib.correct_nan(debug = False, plot = False)
calib.correct_bad_pixels(debug = False, plot = False)
calib.first_frames_removal(debug = False, plot = False)
calib.get_stellar_psf(debug = False, plot = False)
calib.subtract_sky(debug = False, plot = False)
